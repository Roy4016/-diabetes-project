#!/usr/bin/env python3
"""Recheck LR, RF, and XGBoost variants on the current UCI folds.

The saved CCS/MEPS OOF file defines the exact 99,343-row patient-disjoint split.
All model selection and oversampling stay inside each outer training fold.
Optional CCS replacement exchanges diag_1/2/3 for AHRQ CCS categories.
Optional CCS augmentation retains all 46 raw predictors and appends 3 CCS fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diabetes_project.features import make_preprocessor  # noqa: E402
from diabetes_project.metrics import best_f1_threshold, binary_metrics  # noqa: E402
from diabetes_project.paper_comparison import load_comparison_cohort  # noqa: E402
from run_meps_hgb import normalize, raw_discharge_predictors  # noqa: E402
from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs  # noqa: E402
from train_xgboost_ahrq_ccs_prefix import load_ccs  # noqa: E402

METHODS = (
    "lr_standard", "lr_balanced", "rf_standard", "rf_balanced",
    "xgb_standard", "xgb_weighted", "xgb_smotenc_020", "xgb_smotenc_035",
    "xgb_smotenc_050", "xgb_smotenc_100",
)
RATIOS = {"xgb_smotenc_020": .20, "xgb_smotenc_035": .35,
          "xgb_smotenc_050": .50, "xgb_smotenc_100": 1.0}


def read_split(input_file: Path, reference: Path):
    cohort = load_comparison_cohort(input_file)
    table = pd.read_csv(reference, usecols=["encounter_id", "patient_nbr", "y_true", "Fold"])
    if len(cohort) != 99343 or len(table) != len(cohort):
        raise ValueError("Expected the current 99,343-row UCI cohort and OOF file")
    if cohort.encounter_id.duplicated().any() or table.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs")
    lookup = table.set_index("encounter_id")
    aligned = lookup.reindex(cohort.encounter_id.to_numpy())
    if aligned.isna().any().any():
        raise ValueError("OOF and raw UCI cohorts differ")
    if not np.array_equal(aligned.patient_nbr.to_numpy(), cohort.patient_nbr.to_numpy()):
        raise ValueError("Patient IDs differ from the reference split")
    if not np.array_equal(aligned.y_true.to_numpy(), cohort.target.to_numpy()):
        raise ValueError("Labels differ from the reference split")
    fold = aligned.Fold.to_numpy(dtype=int)
    if set(fold) != {1, 2, 3, 4, 5} or pd.DataFrame({
        "patient": cohort.patient_nbr, "fold": fold,
    }).groupby("patient").fold.nunique().max() != 1:
        raise ValueError("Reference split is not five-fold patient-disjoint")
    return cohort, fold


def smotenc_resample(frame, y, columns, categorical, numeric, ratio, seed):
    """Use training-only category codes; decode before the original one-hot fit."""
    encoded = frame.loc[:, columns].copy()
    medians = {}
    levels = {}
    for col in numeric:
        values = pd.to_numeric(encoded[col], errors="coerce")
        median = values.median()
        medians[col] = float(median) if pd.notna(median) else 0.0
        encoded[col] = values.fillna(medians[col]).astype("float64")
    for col in categorical:
        values = encoded[col].astype(object).where(encoded[col].notna(), "__MISSING__").astype(str)
        levels[col] = pd.Index(values.unique()).tolist()
        encoded[col] = pd.Categorical(values, categories=levels[col]).codes.astype("int32")
    current = float(np.sum(y) / np.sum(y == 0))
    if ratio <= current:
        raise ValueError(f"SMOTENC ratio {ratio} must exceed observed ratio {current:.4f}")
    sampler = SMOTENC(
        categorical_features=[columns.index(col) for col in categorical],
        sampling_strategy=ratio, k_neighbors=5, random_state=seed,
    )
    sampled, labels = sampler.fit_resample(encoded[columns], y)
    sampled = pd.DataFrame(sampled, columns=columns)
    for col in numeric:
        sampled[col] = pd.to_numeric(sampled[col], errors="raise").astype(float)
    for col in categorical:
        codes = np.rint(pd.to_numeric(sampled[col], errors="raise").to_numpy(dtype=float)).astype(int)
        if codes.min() < 0 or codes.max() >= len(levels[col]):
            raise ValueError(f"Invalid generated category code: {col}")
        sampled[col] = pd.Series(pd.Categorical.from_codes(codes, levels[col])).astype(object)
    return sampled, np.asarray(labels, dtype=int)


def matrix(preprocessor, frame, columns):
    x = preprocessor.transform(frame.loc[:, columns])
    return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)


def xgb(args, trees, scale_pos_weight):
    return XGBClassifier(
        n_estimators=trees, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=.8, colsample_bytree=.8,
        min_child_weight=3, reg_lambda=1, objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist", device=args.device,
        random_state=args.seed, n_jobs=args.n_jobs,
        scale_pos_weight=scale_pos_weight,
    )


def estimator(args, method, trees=None, scale_pos_weight=1.0):
    if method.startswith("lr_"):
        return LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=args.lr_max_iter,
            class_weight="balanced" if method == "lr_balanced" else None,
            random_state=args.seed,
        )
    if method.startswith("rf_"):
        return RandomForestClassifier(
            n_estimators=args.rf_estimators, max_depth=args.rf_max_depth,
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced" if method == "rf_balanced" else None,
            random_state=args.seed, n_jobs=args.n_jobs,
        )
    return xgb(args, trees, scale_pos_weight)


def run(args):
    cohort, folds = read_split(args.input, args.reference_oof)
    columns = raw_discharge_predictors(cohort)
    if args.diagnosis_representation in ("ccs-replace", "ccs-append"):
        ccs_map, _ = load_ccs(args.ccs_file)
        cohort = attach_uci_ccs(cohort, ccs_map)
        if args.diagnosis_representation == "ccs-replace":
            columns = [f"ccs_{column}" if column in DIAG else column for column in columns]
        else:
            columns += [f"ccs_{column}" for column in DIAG]
        expected = 49 if args.diagnosis_representation == "ccs-append" else 46
        if len(columns) != len(set(columns)) or len(columns) != expected:
            raise ValueError(f"Expected {expected} unique predictors")
    raw, categorical, numeric = normalize(cohort, columns)
    # The shared normalizer identifies categorical ID fields; keep that type
    # in the one-hot preprocessor as well.
    for col in categorical:
        raw[col] = raw[col].astype(object)
    labels = cohort.target.to_numpy(dtype=int)
    patients = cohort.patient_nbr.to_numpy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results, predictions, selections = [], [], []
    for fold in range(1, args.max_folds + 1):
        outer = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        inner_rel, val_rel = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(raw.iloc[outer], labels[outer], groups=patients[outer]))
        inner, val = outer[inner_rel], outer[val_rel]
        if set(patients[inner]) & set(patients[val]) or set(patients[outer]) & set(patients[test]):
            raise ValueError("Patient overlap in split")
        print(f"[Fold {fold}] train={len(outer):,} inner={len(inner):,} val={len(val):,} test={len(test):,}", flush=True)
        # Fit one-hot vocabularies and imputation only on original training rows.
        inner_prep = make_preprocessor(raw.iloc[inner], columns)
        inner_prep.fit(raw.iloc[inner])
        outer_prep = make_preprocessor(raw.iloc[outer], columns)
        outer_prep.fit(raw.iloc[outer])
        x_inner_original = matrix(inner_prep, raw.iloc[inner], columns)
        x_outer_original = matrix(outer_prep, raw.iloc[outer], columns)
        x_val = matrix(inner_prep, raw.iloc[val], columns)
        x_test = matrix(outer_prep, raw.iloc[test], columns)
        for method in args.methods:
            ratio = RATIOS.get(method)
            if ratio is None:
                inner_y, outer_y = labels[inner], labels[outer]
                x_inner, x_outer = x_inner_original, x_outer_original
            else:
                inner_frame, inner_y = smotenc_resample(
                    raw.iloc[inner], labels[inner], columns, categorical, numeric, ratio, args.seed + fold,
                )
                outer_frame, outer_y = smotenc_resample(
                    raw.iloc[outer], labels[outer], columns, categorical, numeric, ratio, args.seed + fold,
                )
                x_inner = matrix(inner_prep, inner_frame, columns)
                x_outer = matrix(outer_prep, outer_frame, columns)
            inner_weight = float(np.sum(labels[inner] == 0) / np.sum(labels[inner])) if method == "xgb_weighted" else 1.0
            is_xgb = method not in METHODS[:4]
            tuning = estimator(args, method, args.n_estimators, inner_weight)
            if is_xgb:
                tuning.set_params(early_stopping_rounds=args.early_stopping_rounds)
                tuning.fit(x_inner, inner_y, eval_set=[(x_val, labels[val])], verbose=False)
            else:
                tuning.fit(x_inner, inner_y)
            val_prob = tuning.predict_proba(x_val)[:, 1]
            threshold = best_f1_threshold(labels[val], val_prob)
            best_trees = int(tuning.best_iteration) + 1 if is_xgb else None
            outer_weight = float(np.sum(labels[outer] == 0) / np.sum(labels[outer])) if method == "xgb_weighted" else 1.0
            final = estimator(args, method, best_trees, outer_weight)
            if is_xgb:
                final.fit(x_outer, outer_y, verbose=False)
            else:
                final.fit(x_outer, outer_y)
            test_prob = final.predict_proba(x_test)[:, 1]
            selections.append({"Fold": fold, "Model": method, "BestTrees": best_trees,
                               "ValidationThreshold": threshold, "TrainRowsAfterSampling": len(outer_y),
                               "ScalePosWeight": outer_weight})
            for policy, cutoff in (("val_f1_optimal", threshold), ("default_0.5", .5)):
                results.append({"Fold": fold, "Model": method, "ThresholdPolicy": policy,
                                **binary_metrics(labels[test], test_prob, cutoff)})
            top_count = max(1, int(np.floor(.10 * len(test))))
            top_prediction = np.zeros(len(test), dtype=int)
            top_prediction[np.argsort(-test_prob, kind="mergesort")[:top_count]] = 1
            top_metrics = binary_metrics(labels[test], test_prob, .5)
            top_metrics.update({
                "Precision": precision_score(labels[test], top_prediction, zero_division=0),
                "Recall": recall_score(labels[test], top_prediction, zero_division=0),
                "F1": f1_score(labels[test], top_prediction, zero_division=0),
                "Accuracy": float(np.mean(labels[test] == top_prediction)),
                "TP": int(np.sum((labels[test] == 1) & (top_prediction == 1))),
                "FP": int(np.sum((labels[test] == 0) & (top_prediction == 1))),
                "TN": int(np.sum((labels[test] == 0) & (top_prediction == 0))),
                "FN": int(np.sum((labels[test] == 1) & (top_prediction == 0))),
                "threshold": float("nan"),
            })
            results.append({"Fold": fold, "Model": method,
                            "ThresholdPolicy": "top_10pct", **top_metrics})
            predictions.append(pd.DataFrame({
                "encounter_id": cohort.encounter_id.iloc[test].to_numpy(),
                "patient_nbr": patients[test], "y_true": labels[test], "Fold": fold,
                "Model": method, "probability": test_prob,
            }))
            print(f"[Fold {fold}] {method:13s} AP={top_metrics['PR_AUC_AP']:.6f} "
                  f"AUROC={top_metrics['AUROC']:.6f}" +
                  (f" trees={best_trees}" if is_xgb else ""), flush=True)
        pd.DataFrame(results).to_csv(args.outdir / "fold_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(args.outdir / "oof_long.csv", index=False)
        pd.DataFrame(selections).to_csv(args.outdir / "selections.csv", index=False)
    metrics = pd.DataFrame(results)
    summary = metrics.groupby(["Model", "ThresholdPolicy"], as_index=False)[
        ["PR_AUC_AP", "AUROC", "Precision", "Recall", "F1", "Brier"]
    ].mean().rename(columns={"PR_AUC_AP": "PR_AUC_mean", "AUROC": "AUROC_mean",
                       "Precision": "Precision_mean", "Recall": "Recall_mean",
                       "F1": "F1_mean", "Brier": "Brier_mean"})
    ap_std = metrics.groupby(["Model", "ThresholdPolicy"])["PR_AUC_AP"].std().rename("PR_AUC_std")
    summary = summary.join(ap_std, on=["Model", "ThresholdPolicy"])
    pooled = pd.concat(predictions, ignore_index=True)
    from sklearn.metrics import average_precision_score
    pooled_ap = pd.Series({
        name: average_precision_score(part.y_true, part.probability)
        for name, part in pooled.groupby("Model")
    }, name="Pooled_PR_AUC")
    summary = summary.join(pooled_ap, on="Model")
    summary.to_csv(args.outdir / "summary.csv", index=False)
    (args.outdir / "run.json").write_text(json.dumps({
        "cohort_rows": len(cohort), "patients": int(cohort.patient_nbr.nunique()),
        "folds_completed": args.max_folds, "seed": args.seed, "methods": args.methods,
        "features": columns, "reference_oof": str(args.reference_oof.resolve()),
        "device": args.device, "validation": "patient-disjoint 20% of outer train",
        "diagnosis_representation": args.diagnosis_representation,
        "ccs_file": str(args.ccs_file.resolve()) if args.diagnosis_representation != "raw" else None,
        "selection": "XGBoost: validation AUCPR early stopping; all models: validation F1 threshold; refit full outer train",
        "threshold_policies": ["val_f1_optimal", "default_0.5", "top_10pct"],
        "primary_metric": "sklearn average_precision_score",
    }, indent=2), encoding="utf-8")
    print(summary.round(6).to_string(index=False), flush=True)
    print(f"Saved to {args.outdir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--diagnosis-representation", choices=("raw", "ccs-replace", "ccs-append"), default="raw")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-folds", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--lr-max-iter", type=int, default=1500)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.outdir is None:
        directory = {"raw": "classic_5fold_recheck_seed777",
                     "ccs-replace": "classic_ccs_replacement_5fold_seed777",
                     "ccs-append": "classic_ccs_augmentation_5fold_seed777"}[args.diagnosis_representation]
        args.outdir = ROOT / "results" / directory
    run(args)


if __name__ == "__main__":
    main()
