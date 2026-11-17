#!/usr/bin/env python3
"""Apply a validation-gated MEPS event-CCS ranking correction to CCS-HGB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from _paths import CCS_RAW, UCI_RAW, result_dir
from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs, make_event_group_rates
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs


WEIGHTS = {"diag_1": 0.5, "diag_2": 0.3, "diag_3": 0.2}
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
FIXED_HGB = {
    "learning_rate": 0.05, "max_iter": 160, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "l2_regularization": 0.0,
}


def meps_rank_score(frame, rates, global_rate):
    """MEPS relative log odds for primary and secondary CCS diagnoses."""
    prior = rates.set_index("ccs").next_er.to_dict()
    center = logit(np.clip(global_rate, 1e-5, 1 - 1e-5))
    score = np.zeros(len(frame), dtype=float)
    covered = np.zeros(len(frame), dtype=bool)
    for diag in DIAG:
        group_rate = frame[f"ccs_{diag}"].map(prior).to_numpy(dtype=float)
        known = np.isfinite(group_rate)
        covered |= known
        contrast = np.zeros(len(frame), dtype=float)
        contrast[known] = logit(np.clip(group_rate[known], 1e-5, 1 - 1e-5)) - center
        score += WEIGHTS[diag] * contrast
    return np.clip(score, -2.0, 2.0), covered


def corrected_probability(base, score, strength):
    return expit(logit(np.clip(base, 1e-6, 1 - 1e-6)) + strength * score)


def reference_predictions(path, hospital, folds, seed):
    reference = pd.read_csv(path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_hgb_tuned",
        "p_ccs_only_hgb", "p_ccs_meps_group_hgb",
    ])
    if reference.encounter_id.duplicated().any() or hospital.encounter_id.duplicated().any():
        raise ValueError("Reference or hospital has duplicate encounter IDs")
    if len(reference) != len(hospital):
        raise ValueError("Reference OOF has a different number of encounters")
    by_encounter = reference.set_index("encounter_id")
    if not by_encounter.index.is_unique:
        raise ValueError("Reference encounter IDs are not unique")
    aligned = by_encounter.reindex(hospital.encounter_id.to_numpy())
    if aligned.isna().any().any():
        raise ValueError("Reference OOF has missing encounters or predictions")
    if not np.array_equal(aligned.patient_nbr.to_numpy(), hospital.patient_nbr.to_numpy()):
        raise ValueError("Reference patient IDs differ")
    if not np.array_equal(aligned.y_true.to_numpy(), hospital.target.to_numpy()):
        raise ValueError("Reference labels differ")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test_ids) in enumerate(splitter.split(hospital, hospital.target, hospital.patient_nbr), 1):
        if not np.all(aligned.Fold.to_numpy()[test_ids] == fold):
            raise ValueError("Reference fold assignment differs; use matching --folds and --seed")
    return aligned.reset_index(drop=True)


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    columns = raw_discharge_predictors(hospital)
    ccs_map, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, ccs_map)
    columns += [f"ccs_{diag}" for diag in DIAG]
    reference = reference_predictions(args.reference_oof, hospital, args.folds, args.seed)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, n_events, n_linked = make_event_group_rates(
        events, event_ccs, args.min_group, args.alpha,
    )
    rates.to_csv(output / "meps_event_ccs_rates.csv", index=False)
    print(f"[MEPS] {n_events:,} events, {n_linked:,} linked CCS records, {len(rates)} groups", flush=True)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    parts, selections = [], []
    for fold, (train_ids, test_ids) in enumerate(
        splitter.split(hospital, hospital.target, hospital.patient_nbr), 1,
    ):
        train, test = hospital.iloc[train_ids], hospital.iloc[test_ids]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient overlap in outer fold")
        inner_ids, val_ids = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=args.seed + fold,
        ).split(train, train.target, train.patient_nbr))
        inner, validation = train.iloc[inner_ids], train.iloc[val_ids]
        x_inner, x_val, _ = encode_frames(inner, validation, validation, columns, args.min_frequency)
        model = hgb_model(args.seed + fold, FIXED_HGB, early_stopping=False)
        model.fit(x_inner, inner.target.to_numpy())
        val_base = model.predict_proba(x_val)[:, 1]
        val_score, val_covered = meps_rank_score(validation, rates, global_rates["next_er"])
        val_ap = {
            strength: average_precision_score(
                validation.target, corrected_probability(val_base, val_score, strength),
            )
            for strength in LAMBDA_GRID
        }
        best = max(LAMBDA_GRID, key=lambda strength: val_ap[strength])
        chosen = best if val_ap[best] >= val_ap[0.0] + args.min_validation_gain else 0.0

        # Match the validation model family when making outer-test predictions.
        x_train, _, x_test = encode_frames(train, train, test, columns, args.min_frequency)
        final_model = hgb_model(args.seed + fold, FIXED_HGB, early_stopping=False)
        final_model.fit(x_train, train.target.to_numpy())
        test_base = final_model.predict_proba(x_test)[:, 1]
        test_score, test_covered = meps_rank_score(test, rates, global_rates["next_er"])
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        part["p_fixed_ccs_hgb"] = test_base
        part["p_meps_rank_fixed_025"] = corrected_probability(test_base, test_score, 0.25)
        part["p_meps_rank_gated"] = corrected_probability(test_base, test_score, chosen)
        part["meps_rank_score"] = test_score
        part["meps_group_covered"] = test_covered
        part["p_ccs_only_hgb"] = reference.iloc[test_ids].p_ccs_only_hgb.to_numpy()
        part["p_ccs_meps_group_hgb"] = reference.iloc[test_ids].p_ccs_meps_group_hgb.to_numpy()
        part["p_hgb_tuned"] = reference.iloc[test_ids].p_hgb_tuned.to_numpy()
        parts.append(part)
        selections.append({"Fold": fold, "SelectedStrength": chosen,
                           "ValidationBaseAP": val_ap[0.0], "ValidationBestAP": val_ap[best],
                           "ValidationMEPSCoverage": float(val_covered.mean()),
                           "TestMEPSCoverage": float(test_covered.mean()),
                           "TestBaseAP": average_precision_score(test.target, test_base),
                           "TestGatedAP": average_precision_score(test.target, part.p_meps_rank_gated)})
        print(f"[Fold {fold}] strength={chosen:.2f} val_gain={val_ap[best]-val_ap[0.0]:+.6f} "
              f"base={selections[-1]['TestBaseAP']:.6f} gated={selections[-1]['TestGatedAP']:.6f}", flush=True)
        pd.concat(parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)

    oof = pd.concat(parts, ignore_index=True)
    models = ["hgb_tuned", "ccs_only_hgb", "ccs_meps_group_hgb", "fixed_ccs_hgb",
              "meps_rank_fixed_025", "meps_rank_gated"]
    metrics = pd.DataFrame([
        {"Fold": int(fold), "Experiment": name,
         "PR_AUC_AP": average_precision_score(part.y_true, part[f"p_{name}"])}
        for fold, part in oof.groupby("Fold", sort=True) for name in models
    ])
    summary = metrics.groupby("Experiment").PR_AUC_AP.agg(["mean", "std"]).reset_index()
    comparisons = [("meps_rank_fixed_025", "fixed_ccs_hgb"),
                   ("meps_rank_gated", "fixed_ccs_hgb"),
                   ("meps_rank_gated", "ccs_only_hgb"),
                   ("meps_rank_gated", "ccs_meps_group_hgb"),
                   ("meps_rank_gated", "hgb_tuned")]
    bootstrap = cluster_bootstrap(oof, models, comparisons, args.bootstrap, args.seed + 17000)
    oof.to_csv(output / "oof_all.csv", index=False)
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "selection_diagnostics.csv", index=False)
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--ccs-file", default=CCS_RAW)
    parser.add_argument("--event-data", default="data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", default="data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--outdir", default=result_dir("meps_event_rank_correction"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--min-validation-gain", type=float, default=0.0002)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if (args.folds < 2 or args.min_group < 1 or args.alpha < 0 or
            args.min_frequency < 1 or args.min_validation_gain < 0 or args.bootstrap < 1):
        parser.error("Invalid fold, group-rate, validation, or bootstrap settings")
    for path in (args.hospital, args.ccs_file, args.event_data, args.event_ccs, args.reference_oof):
        if not Path(path).is_file():
            parser.error(f"Required file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
