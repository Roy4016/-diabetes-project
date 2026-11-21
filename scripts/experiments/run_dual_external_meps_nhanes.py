#!/usr/bin/env python3
"""Compare MEPS+NHANES demographic grouping and external multitask transfer.

The two sources retain separate outcomes: MEPS next-year ER/IP utilization and
NHANES past-year hospitalization. Only external records fit the group statistics
and multitask encoder. The UCI test folds are never used for feature fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.dual_external import (  # noqa: E402
    DEMO, grouping_features, meps_frame, multitask_features, nhanes_frame,
)
from diabetes_project.harmonized_meps import hospital_bridge, load_longitudinal_pairs  # noqa: E402
from diabetes_project.paper_comparison import top_fraction_metrics  # noqa: E402

from run_cdc_encoder_control import prepare_hospital  # noqa: E402
from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs  # noqa: E402
from run_meps_hgb import raw_discharge_predictors  # noqa: E402
from run_nested_hgb_blend import encode_frames, hgb_model  # noqa: E402
from run_paper_fair_comparison import paired_patient_bootstrap  # noqa: E402
from train_xgboost_ahrq_ccs_prefix import load_ccs  # noqa: E402

METHODS = ("grouping", "multitask")
SOURCES = ("meps", "nhanes", "dual")


def load_reference(path: Path) -> pd.DataFrame:
    oof = pd.read_csv(path)
    required = {"encounter_id", "patient_nbr", "y_true", "Fold", "p_ccs_only_hgb",
                "p_ccs_meps_group_hgb"}
    if missing := required - set(oof):
        raise ValueError(f"Reference OOF missing columns: {sorted(missing)}")
    if len(oof) != 99343 or oof.encounter_id.duplicated().any():
        raise ValueError("Expected the fixed 99,343-encounter latest OOF")
    if set(oof.Fold) != {1, 2, 3, 4, 5} or oof.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Reference OOF does not have five patient-disjoint folds")
    return oof


def hospital_frame(args, oof):
    hospital, _ = prepare_hospital(args.hospital)
    predictors = raw_discharge_predictors(hospital)
    ccs_map, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, ccs_map)
    if hospital.encounter_id.duplicated().any() or len(hospital) != len(oof):
        raise ValueError("Hospital cohort does not match the reference OOF")
    hospital = hospital.set_index("encounter_id").loc[oof.encounter_id].reset_index()
    if not np.array_equal(hospital.patient_nbr.to_numpy(), oof.patient_nbr.to_numpy()) or not np.array_equal(
        hospital.target.to_numpy(), oof.y_true.to_numpy()):
        raise ValueError("Hospital patients or labels differ from the reference OOF")
    return hospital, predictors + [f"ccs_{name}" for name in DIAG]


def selected_hgb(path: Path) -> pd.DataFrame:
    choice = pd.read_csv(path)
    choice = choice.loc[choice.Experiment.eq("ccs_only_hgb")]
    if set(choice.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected CCS-only selected HGB parameters in all five folds")
    return choice.set_index("Fold")


def align_catboost(oof: pd.DataFrame, path: Path) -> np.ndarray:
    saved = pd.read_csv(path, usecols=["encounter_id", "patient_nbr", "y_true", "Fold",
                                       "catboost_raw"])
    if saved.encounter_id.duplicated().any() or len(saved) != len(oof):
        raise ValueError("CatBoost OOF has a different cohort")
    joined = oof[["encounter_id", "patient_nbr", "y_true", "Fold"]].merge(
        saved, on="encounter_id", how="left", validate="one_to_one", suffixes=("", "_saved"))
    if joined.catboost_raw.isna().any() or any(
        not joined[name].eq(joined[f"{name}_saved"]).all()
        for name in ("patient_nbr", "y_true", "Fold")
    ):
        raise ValueError("CatBoost predictions are not on the same patients, labels and folds")
    return joined.catboost_raw.to_numpy(dtype=float)


def fit_fold(frame, columns, train, test, choice, fold, args):
    x_train, _, x_test = encode_frames(frame.loc[train], frame.loc[train],
                                      frame.loc[test], columns, args.min_frequency)
    params = json.loads(choice.loc[fold, "Parameters"])
    params["max_iter"] = int(choice.loc[fold, "Iterations"])
    model = hgb_model(args.seed + fold, params, early_stopping=False)
    model.fit(x_train, frame.loc[train, "target"].to_numpy(dtype=int))
    return model.predict_proba(x_test)[:, 1]


def report(oof, names, outdir, args):
    rows = []
    for fold, part in oof.groupby("Fold", sort=True):
        for name in names:
            rows.append({"Fold": int(fold), "Experiment": name,
                         "Test_Rows": len(part), "Test_Patients": int(part.patient_nbr.nunique()),
                         **top_fraction_metrics(part.y_true, part[f"p_{name}"], args.alert_fraction)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "fold_metrics.csv", index=False)
    summary = metrics.groupby("Experiment", sort=False).agg(
        Folds=("Fold", "size"), PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"), AUROC_mean=("AUROC", "mean"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        F1_mean=("F1_score", "mean"), Brier_mean=("Brier_Score", "mean"),
    ).reset_index()
    summary["Pooled_PR_AUC"] = [average_precision_score(oof.y_true, oof[f"p_{name}"])
                                 for name in summary.Experiment]
    summary.to_csv(outdir / "summary.csv", index=False)
    print("\nOuter-fold summary\n", summary.round(6).to_string(index=False), flush=True)
    pairs = []
    for method in args.methods:
        dual = f"{method}_dual"
        pairs += [(dual, "ccs_only_hgb"), (dual, "meps_event_ccs_hgb"),
                  (dual, f"{method}_meps"), (dual, f"{method}_nhanes"),
                  (f"{dual}_catboost_raw50", "ccs_only_catboost_raw50"),
                  (f"{dual}_catboost_raw50", "meps_event_ccs_catboost_raw50"),
                  (f"{dual}_catboost_raw50", f"{method}_meps_catboost_raw50"),
                  (f"{dual}_catboost_raw50", f"{method}_nhanes_catboost_raw50")]
    if set(args.methods) == set(METHODS):
        pairs.append(("multitask_dual", "grouping_dual"))
    temp = oof.rename(columns={f"p_{name}": name for name in names})
    paired_patient_bootstrap(temp, pairs, outdir, args.bootstrap, args.seed)


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    oof = load_reference(args.reference_oof)
    catboost_raw = align_catboost(oof, args.catboost_oof)
    hospital, columns = hospital_frame(args, oof)
    choices = selected_hgb(args.selected_hgb)
    pairs = meps_frame(load_longitudinal_pairs(args.meps))
    nhanes = nhanes_frame(pd.read_csv(args.nhanes))
    demo = hospital_bridge(hospital).loc[:, list(DEMO)].reset_index(drop=True)
    print(f"[Sources] MEPS={len(pairs):,} longitudinal pairs, NHANES={len(nhanes):,} respondents; "
          f"UCI={len(oof):,} encounters", flush=True)

    feature_sets = {}
    if "grouping" in args.methods:
        feature_sets["grouping"] = grouping_features(demo, pairs, nhanes, args.group_alpha)
    if "multitask" in args.methods:
        feature_sets["multitask"], training_log = multitask_features(
            demo, pairs, nhanes, args.epochs, args.batch_size, args.seed, args.torch_threads)
        pd.DataFrame(training_log).to_csv(args.outdir / "external_training.csv", index=False)
    pd.DataFrame([
        {"Source": "MEPS", "Rows": len(pairs), "Patients": pairs.person_key.nunique(),
         "ER_Prevalence": pairs.meps_er.mean(), "Hospital_Prevalence": pairs.meps_ip.mean()},
        {"Source": "NHANES", "Rows": len(nhanes), "Patients": nhanes.person_key.nunique(),
         "ER_Prevalence": np.nan, "Hospital_Prevalence": nhanes.nhanes_hosp.mean()},
    ]).to_csv(args.outdir / "external_sources.csv", index=False)

    for method, versions in feature_sets.items():
        for source, features in versions.items():
            if len(features) != len(hospital) or features.isna().any().any():
                raise ValueError(f"{method}/{source} produced incomplete UCI features")
            for name in features:
                hospital[f"{method}_{source}_{name}"] = features[name].to_numpy(dtype=np.float32)

    oof = oof[["encounter_id", "patient_nbr", "y_true", "Fold",
               "p_ccs_only_hgb", "p_ccs_meps_group_hgb"]].rename(
                   columns={"p_ccs_meps_group_hgb": "p_meps_event_ccs_hgb"})
    oof["p_catboost_raw"] = catboost_raw
    oof["p_ccs_only_catboost_raw50"] = (oof.p_ccs_only_hgb + catboost_raw) / 2
    oof["p_meps_event_ccs_catboost_raw50"] = (oof.p_meps_event_ccs_hgb + catboost_raw) / 2
    names = ["ccs_only_hgb", "meps_event_ccs_hgb", "catboost_raw",
             "ccs_only_catboost_raw50", "meps_event_ccs_catboost_raw50"]
    names += [f"{method}_{source}" for method in args.methods for source in SOURCES]
    names += [f"{method}_{source}_catboost_raw50"
              for method in args.methods for source in SOURCES]
    discrepancies = {}
    folds = list(range(1, args.max_folds + 1))
    for fold in folds:
        test = oof.Fold.eq(fold).to_numpy()
        train = ~test
        if fold == 1:
            baseline = fit_fold(hospital, columns, train, test, choices, fold, args)
            discrepancies["1"] = float(np.max(np.abs(baseline - oof.loc[test, "p_ccs_only_hgb"])))
            if discrepancies["1"] > 1e-8:
                raise ValueError(f"CCS-only HGB refit does not match saved OOF: {discrepancies['1']}")
            print("[Audit] Fold 1 CCS-only HGB refit exactly matches saved OOF", flush=True)
        for method in args.methods:
            for source in SOURCES:
                feature_names = [f"{method}_{source}_{name}" for name in feature_sets[method][source]]
                name = f"{method}_{source}"
                probability = fit_fold(hospital, columns + feature_names,
                                       train, test, choices, fold, args)
                oof.loc[test, f"p_{name}"] = probability
                oof.loc[test, f"p_{name}_catboost_raw50"] = (
                    probability + oof.loc[test, "p_catboost_raw"].to_numpy()) / 2
                ap = average_precision_score(oof.loc[test, "y_true"], probability)
                print(f"[Fold {fold}] {name} PR_AUC={ap:.6f}", flush=True)
        oof.loc[test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)
    completed = oof.loc[oof.Fold.isin(folds)].copy()
    if completed[[f"p_{name}" for name in names]].isna().any().any():
        raise ValueError("Incomplete OOF predictions")
    completed.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    report(completed, names, args.outdir, args)
    config = {**{key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
              "completed_folds": folds, "baseline_refit_max_abs_difference": discrepancies,
              "external_label_semantics": {
                  "MEPS": "next-year ER and inpatient utilization from consecutive annual pairs",
                  "NHANES": "past-year hospitalization among diabetic respondents",
                  "UCI": "30-day hospital readmission"},
              "selection": "CCS-only HGB parameters selected inside each original outer train fold; "
                           "fixed for every new variant; no outer test label tuning",
              "grouping": "source-specific age/sex/race hierarchical smoothed rates; "
                          "dual combines source-specific logit lifts, not raw labels",
              "multitask": "external-only shared demographic encoder with separate MEPS ER/IP "
                           "and NHANES hospitalization heads",
              "blend": "fixed 50/50 mean of each HGB and the audited CatBoost raw-ICD OOF"}
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--meps", type=Path,
                        default=ROOT / "data/processed/meps_2008_2024/meps_diabetes_bridge8.csv")
    parser.add_argument("--nhanes", type=Path,
                        default=ROOT / "data/processed/nhanes/diabetes_hospitalization_2015_2018.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--reference-oof", type=Path,
                        default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--selected-hgb", type=Path,
                        default=ROOT / "results/meps_event_ccs_group_hgb_seed777/selected_hgb.csv")
    parser.add_argument("--catboost-oof", type=Path,
                        default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/meps_nhanes_dual_external_seed777")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--group-alpha", type=float, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if (not 1 <= args.max_folds <= 5 or args.min_frequency < 1 or args.group_alpha <= 0 or
            args.epochs < 1 or args.batch_size < 1 or args.torch_threads < 1 or
            args.bootstrap < 0 or not 0 < args.alert_fraction < 1 or
            len(set(args.methods)) != len(args.methods)):
        parser.error("Invalid folds, hyperparameters, bootstrap or duplicate methods")
    for path in (args.hospital, args.meps, args.nhanes, args.ccs_file,
                 args.reference_oof, args.selected_hgb, args.catboost_oof):
        if not path.is_file():
            parser.error(f"Required input file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
