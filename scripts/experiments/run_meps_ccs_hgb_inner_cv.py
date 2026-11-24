#!/usr/bin/env python3
"""Compare CCS-only and MEPS event-CCS HGB with 3-fold inner group CV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diabetes_project.paper_comparison import top_fraction_metrics

from _paths import CCS_RAW, UCI_RAW, result_dir
from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_ccs_group_hgb import (
    DIAG, EVENT_STATS, attach_group_features, attach_uci_ccs, make_event_group_rates,
)
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import HGB_GRID, encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs


VARIANTS = {"ccs_only_inner_cv": [], "ccs_meps_inner_cv": list(EVENT_STATS)}


def choose_by_inner_cv(train, columns, folds, min_frequency, seed):
    """Choose HGB settings and iteration by mean patient-disjoint AP."""
    splits = list(StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed,
    ).split(train, train.target, train.patient_nbr))
    encoded = []
    for fit_idx, val_idx in splits:
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        if set(fit.patient_nbr).intersection(val.patient_nbr):
            raise ValueError("Patient overlap in inner CV")
        if fit.target.nunique() < 2 or val.target.nunique() < 2:
            raise ValueError("One-class partition in inner CV")
        x_fit, x_val, _ = encode_frames(fit, val, val, columns, min_frequency)
        encoded.append((x_fit, x_val, fit.target.to_numpy(), val.target.to_numpy()))
    results = []
    for number, params in enumerate(HGB_GRID, 1):
        checkpoints = list(range(20, params["max_iter"] + 1, 10))
        if checkpoints[-1] != params["max_iter"]:
            checkpoints.append(params["max_iter"])
        fold_curves = []
        for inner_fold, (x_fit, x_val, y_fit, y_val) in enumerate(encoded, 1):
            model = hgb_model(seed + number * 101 + inner_fold, params, early_stopping=False)
            model.fit(x_fit, y_fit)
            curve = []
            for iteration, probability in enumerate(model.staged_predict_proba(x_val), 1):
                if iteration in checkpoints:
                    curve.append(float(average_precision_score(y_val, probability[:, 1])))
            if len(curve) != len(checkpoints):
                raise ValueError("HGB staged predictions did not cover all checkpoints")
            fold_curves.append(curve)
        mean_curve = np.mean(fold_curves, axis=0)
        best_index = int(np.argmax(mean_curve))
        best_iteration = checkpoints[best_index]
        scores = [float(curve[best_index]) for curve in fold_curves]
        results.append({"candidate": number, "params": params,
                        "iterations": best_iteration, "mean_ap": float(mean_curve[best_index]),
                        "fold_aps": scores})
        print(f"  [HGB {number}/{len(HGB_GRID)}] inner_AP={mean_curve[best_index]:.6f} "
              f"iter={best_iteration} folds={[round(score, 6) for score in scores]}", flush=True)
    best = max(results, key=lambda item: item["mean_ap"])
    return best, results


def align_reference(oof, path):
    reference = pd.read_csv(path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_hgb_tuned",
        "p_ccs_only_hgb", "p_ccs_meps_group_hgb",
    ])
    if reference.encounter_id.duplicated().any() or oof.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs")
    aligned = oof.merge(reference, on="encounter_id", how="left", validate="one_to_one", suffixes=("", "_ref"))
    if len(aligned) != len(reference) or aligned.p_ccs_only_hgb.isna().any():
        raise ValueError("Reference OOF does not cover current encounters")
    for name in ("patient_nbr", "Fold", "y_true"):
        if not np.array_equal(aligned[name].to_numpy(), aligned[f"{name}_ref"].to_numpy()):
            raise ValueError(f"Reference {name} mismatch")
    return aligned.drop(columns=[f"{name}_ref" for name in ("patient_nbr", "Fold", "y_true")])


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    base_columns = raw_discharge_predictors(hospital)
    mapping, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, mapping)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, n_events, n_linked = make_event_group_rates(
        events, event_ccs, args.min_group, args.alpha,
    )
    hospital = attach_group_features(hospital, rates, global_rates, EVENT_STATS)
    rates.to_csv(output / "meps_event_ccs_rates.csv", index=False)
    print(f"[MEPS] {n_events:,} events, {n_linked:,} CCS records, {len(rates)} groups", flush=True)
    base_columns += [f"ccs_{diag}" for diag in DIAG]
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    reference = pd.read_csv(args.reference_oof, usecols=["encounter_id", "patient_nbr", "Fold", "y_true"])
    if len(reference) != len(hospital) or reference.encounter_id.duplicated().any():
        raise ValueError("Reference OOF cohort differs from hospital cohort")
    reference = reference.set_index("encounter_id").reindex(hospital.encounter_id.to_numpy())
    if reference.Fold.isna().any() or not np.array_equal(reference.patient_nbr.to_numpy(), hospital.patient_nbr.to_numpy()) or not np.array_equal(reference.y_true.to_numpy(), hospital.target.to_numpy()):
        raise ValueError("Reference encounter, patient or label mismatch")
    splits = list(splitter.split(hospital, hospital.target, hospital.patient_nbr))
    audit = {"Rows": len(hospital), "Patients": int(hospital.patient_nbr.nunique()),
             "Seed": args.seed, "Folds": []}
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        if set(hospital.patient_nbr.iloc[train_idx]) & set(hospital.patient_nbr.iloc[test_idx]):
            raise ValueError(f"Patient overlap in fold {fold}")
        if not np.all(reference.Fold.to_numpy()[test_idx] == fold):
            raise ValueError(f"Reference outer fold {fold} mismatch")
        audit["Folds"].append({"Fold": fold, "Test_Rows": len(test_idx),
                               "Test_Patients": int(hospital.patient_nbr.iloc[test_idx].nunique())})
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("[Audit] exact reference five-fold assignment and patient disjointness verified", flush=True)
    if args.audit_only:
        return
    parts, selection_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train, test = hospital.iloc[train_idx], hospital.iloc[test_idx]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient overlap in outer fold")
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        for name, extra in VARIANTS.items():
            columns = base_columns + extra
            print(f"[Fold {fold}] {name}: 3-fold inner validation", flush=True)
            best, candidates = choose_by_inner_cv(
                train, columns, args.inner_folds, args.min_frequency, args.seed + fold * 1000,
            )
            x_train, _, x_test = encode_frames(train, train, test, columns, args.min_frequency)
            model = hgb_model(args.seed + fold,
                              {**best["params"], "max_iter": best["iterations"]},
                              early_stopping=False)
            model.fit(x_train, train.target.to_numpy())
            prediction = model.predict_proba(x_test)[:, 1]
            part[f"p_{name}"] = prediction
            selection_rows.append({"Fold": fold, "Experiment": name,
                                   "Candidate": best["candidate"], "InnerMeanAP": best["mean_ap"],
                                   "Iterations": best["iterations"],
                                   "Parameters": json.dumps(best["params"]),
                                   "Candidates": json.dumps(candidates)})
            print(f"[Fold {fold}] {name} PR_AUC={average_precision_score(test.target, prediction):.6f}", flush=True)
        parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)
    oof = align_reference(pd.concat(parts, ignore_index=True), args.reference_oof)
    models = ["hgb_tuned", "ccs_only_hgb", "ccs_meps_group_hgb", *VARIANTS]
    metrics = pd.DataFrame([
        {"Fold": int(fold), "Experiment": name, "N_encounters": len(group),
         "N_patients": int(group.patient_nbr.nunique()),
         **top_fraction_metrics(group.y_true, group[f"p_{name}"], args.alert_fraction)}
        for fold, group in oof.groupby("Fold", sort=True) for name in models
    ])
    summary = metrics.groupby("Experiment", sort=False).agg(
        Folds=("Fold", "size"), PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"), AUROC_mean=("AUROC", "mean"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        F1_mean=("F1_score", "mean"), Brier_mean=("Brier_Score", "mean"),
    ).reset_index()
    summary["Pooled_PR_AUC"] = [average_precision_score(oof.y_true, oof[f"p_{name}"])
                                 for name in summary.Experiment]
    comparisons = [("ccs_meps_inner_cv", "ccs_only_inner_cv"),
                   ("ccs_meps_inner_cv", "ccs_meps_group_hgb"),
                   ("ccs_meps_inner_cv", "ccs_only_hgb"),
                   ("ccs_only_inner_cv", "ccs_only_hgb")]
    bootstrap = (cluster_bootstrap(oof, models, comparisons, args.bootstrap, args.seed + 17000)
                 if args.bootstrap else None)
    oof.to_csv(output / "oof_all.csv", index=False)
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if bootstrap is not None:
        bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "inner_cv_selection.csv", index=False)
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    if bootstrap is not None:
        print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--ccs-file", default=CCS_RAW)
    parser.add_argument("--event-data", default="data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", default="data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--reference-oof", default=str(ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv"))
    parser.add_argument("--outdir", default=result_dir("meps_ccs_hgb_inner_cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--audit-only", action="store_true", help="Verify input and five-fold assignment without fitting HGB")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if (args.folds != 5 or args.inner_folds < 2 or args.min_group < 1 or
            args.alpha < 0 or args.min_frequency < 1 or args.bootstrap < 0 or
            not 0 < args.alert_fraction < 1):
        parser.error("Invalid fold, group-rate, or bootstrap settings")
    for path in (args.hospital, args.ccs_file, args.event_data, args.event_ccs, args.reference_oof):
        if not Path(path).is_file():
            parser.error(f"Required file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
