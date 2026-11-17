#!/usr/bin/env python3
"""Test MEPS event-CCS rates as priors for cross-fitted hospital group risks."""

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
from run_nested_hgb_blend import choose_hgb, encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs


GROUP_COLUMNS = [f"ccs_{name}" for name in DIAG]
RISK_COLUMNS = [f"group_risk_{name}" for name in DIAG] + ["group_risk_mean", "group_risk_max"]
VARIANTS = {"hospital_group_prior": 0.0, "meps_group_prior": 1.0}


def attach_group_risk(source, destination, external_rates, external_global, prior_strength, external_scale):
    """Fit group rates on source labels, then attach predictions to destination."""
    counts = source.groupby("patient_nbr").patient_nbr.transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    y = source.target.to_numpy(dtype=float)
    global_rate = float(np.average(y, weights=weights))
    long = pd.concat([
        pd.DataFrame({"row": np.arange(len(source)), "ccs": source[col].to_numpy(),
                      "weight": weights, "weighted_y": weights * y})
        for col in GROUP_COLUMNS
    ], ignore_index=True).drop_duplicates(["row", "ccs"])
    long = long.loc[long.ccs.notna()]
    groups = long.groupby("ccs").agg(weight=("weight", "sum"), weighted_y=("weighted_y", "sum"))
    external = external_rates.set_index("ccs").next_er.to_dict()
    external_prior = {}
    for ccs in groups.index.union(pd.Index(external)):
        external_rate = float(external.get(ccs, external_global))
        external_prior[ccs] = float(expit(
            logit(np.clip(global_rate, 1e-5, 1 - 1e-5))
            + external_scale * (
                logit(np.clip(external_rate, 1e-5, 1 - 1e-5))
                - logit(np.clip(external_global, 1e-5, 1 - 1e-5))
            )
        ))
    group_prior = pd.Series(external_prior)
    group_weight = groups.weight.reindex(group_prior.index, fill_value=0.0)
    group_sum = groups.weighted_y.reindex(group_prior.index, fill_value=0.0)
    posterior = (group_sum + prior_strength * group_prior) / (group_weight + prior_strength)
    output = destination.copy()
    for diag, ccs_col in zip(DIAG, GROUP_COLUMNS):
        output[f"group_risk_{diag}"] = (
            output[ccs_col].map(posterior).fillna(global_rate).astype(np.float32)
        )
    output["group_risk_mean"] = output[[f"group_risk_{name}" for name in DIAG]].mean(axis=1)
    output["group_risk_max"] = output[[f"group_risk_{name}" for name in DIAG]].max(axis=1)
    return output


def crossfit_group_risk(train, evaluation, external_rates, external_global,
                        prior_strength, external_scale, folds, seed):
    if train.patient_nbr.nunique() < folds:
        raise ValueError("Too few patients for group-risk cross-fitting")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_parts = []
    for fit_idx, held_idx in splitter.split(train, train.target, train.patient_nbr):
        fit = train.iloc[fit_idx]
        held = train.iloc[held_idx]
        if set(fit.patient_nbr).intersection(held.patient_nbr):
            raise ValueError("Patient overlap in group-risk cross-fit")
        part = attach_group_risk(fit, held, external_rates, external_global,
                                 prior_strength, external_scale)
        part["_crossfit_position"] = held_idx
        train_parts.append(part)
    encoded_train = pd.concat(train_parts).sort_values("_crossfit_position").drop(columns="_crossfit_position")
    if len(encoded_train) != len(train):
        raise ValueError("Cross-fit did not cover all training rows")
    encoded_train.index = train.index
    encoded_eval = attach_group_risk(train, evaluation, external_rates, external_global,
                                     prior_strength, external_scale)
    return encoded_train, encoded_eval


def align_reference(oof, reference_path):
    reference = pd.read_csv(reference_path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_ccs_only_hgb", "p_ccs_meps_group_hgb",
    ])
    if oof.encounter_id.duplicated().any() or reference.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs in OOF predictions")
    aligned = oof.merge(reference, on="encounter_id", how="left", validate="one_to_one", suffixes=("", "_ref"))
    if len(aligned) != len(reference) or aligned.p_ccs_only_hgb.isna().any():
        raise ValueError("Reference OOF does not match current encounters")
    for column in ("patient_nbr", "Fold", "y_true"):
        if not np.array_equal(aligned[column].to_numpy(), aligned[f"{column}_ref"].to_numpy()):
            raise ValueError(f"Reference OOF {column} mismatch")
    return aligned.drop(columns=[f"{column}_ref" for column in ("patient_nbr", "Fold", "y_true")])


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    base_columns = raw_discharge_predictors(hospital) + GROUP_COLUMNS
    ccs_map, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, ccs_map)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, n_events, n_linked = make_event_group_rates(
        events, event_ccs, args.min_group, args.alpha,
    )
    rates.to_csv(output / "meps_event_ccs_rates.csv", index=False)
    print(f"[MEPS] {n_events:,} events, {n_linked:,} linked CCS records, {len(rates)} groups", flush=True)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_parts, choices = [], []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(hospital, hospital.target, hospital.patient_nbr), 1,
    ):
        train, test = hospital.iloc[train_idx], hospital.iloc[test_idx]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient overlap in outer split")
        inner_idx, val_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(train, train.target, train.patient_nbr))
        inner, validation = train.iloc[inner_idx], train.iloc[val_idx]
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        for name, scale in VARIANTS.items():
            encoded_inner, encoded_val = crossfit_group_risk(
                inner, validation, rates, global_rates["next_er"],
                args.prior_strength, scale, args.crossfit_folds, args.seed + fold * 100 + 1,
            )
            columns = base_columns + RISK_COLUMNS
            x_inner, x_val, _ = encode_frames(encoded_inner, encoded_val, encoded_val,
                                               columns, args.min_frequency)
            val_ap, params, iterations, _ = choose_hgb(
                x_inner, x_val, inner.target.to_numpy(), validation.target.to_numpy(),
                args.seed + fold,
            )
            encoded_train, encoded_test = crossfit_group_risk(
                train, test, rates, global_rates["next_er"],
                args.prior_strength, scale, args.crossfit_folds, args.seed + fold * 100 + 2,
            )
            x_train, _, x_test = encode_frames(encoded_train, encoded_train, encoded_test,
                                               columns, args.min_frequency)
            model = hgb_model(args.seed + fold, {**params, "max_iter": iterations},
                              early_stopping=False).fit(x_train, train.target.to_numpy())
            probability = model.predict_proba(x_test)[:, 1]
            part[f"p_{name}"] = probability
            choices.append({"Fold": fold, "Experiment": name, "ValidationAP": val_ap,
                            "Iterations": iterations, "Parameters": json.dumps(params)})
            print(f"[Fold {fold}] {name} PR_AUC={average_precision_score(test.target, probability):.6f}", flush=True)
        oof_parts.append(part)
        pd.concat(oof_parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)

    oof = align_reference(pd.concat(oof_parts, ignore_index=True), args.reference_oof)
    models = ["ccs_only_hgb", "ccs_meps_group_hgb", *VARIANTS]
    metrics = pd.DataFrame([
        {"Fold": int(fold), "Experiment": name,
         "PR_AUC_AP": average_precision_score(part.y_true, part[f"p_{name}"])}
        for fold, part in oof.groupby("Fold", sort=True) for name in models
    ])
    summary = metrics.groupby("Experiment").PR_AUC_AP.agg(["mean", "std"]).reset_index()
    comparisons = [("meps_group_prior", "hospital_group_prior"),
                   ("meps_group_prior", "ccs_meps_group_hgb"),
                   ("meps_group_prior", "ccs_only_hgb"),
                   ("hospital_group_prior", "ccs_only_hgb")]
    bootstrap = cluster_bootstrap(oof, models, comparisons, args.bootstrap, args.seed + 17000)
    oof.to_csv(output / "oof_all.csv", index=False)
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "selected_hgb.csv", index=False)
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
    parser.add_argument("--outdir", default=result_dir("meps_group_prior_hgb"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.)
    parser.add_argument("--prior-strength", type=float, default=100.)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if (args.folds < 2 or args.crossfit_folds < 2 or args.min_group < 1
            or args.alpha < 0 or args.prior_strength <= 0 or args.min_frequency < 1
            or args.bootstrap < 1):
        parser.error("Invalid folds, group rate, or bootstrap settings")
    for path in (args.hospital, args.ccs_file, args.event_data, args.event_ccs, args.reference_oof):
        if not Path(path).is_file():
            parser.error(f"Required file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
