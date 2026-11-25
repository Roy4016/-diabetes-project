#!/usr/bin/env python3
"""Compare fixed 50:50 Raw HGB and Raw+CCS HGB ensembles on saved OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analyze_diagnosis_raw_ccs_5fold import KEYS, align, bootstrap_metrics, reference

ROOT = Path(__file__).resolve().parents[2]


def run(args: argparse.Namespace) -> None:
    ref = reference(args.reference_oof)
    source = pd.read_csv(args.oof)
    required = KEYS + ["HGB__Raw", "HGB__Raw_plus_CCS", "CatBoost__Raw_plus_CCS"]
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"OOF missing columns: {missing}")
    raw_hgb = align(ref, source, "HGB__Raw", "Raw HGB")
    augmented_hgb = align(ref, source, "HGB__Raw_plus_CCS", "Raw+CCS HGB")
    augmented_catboost = align(ref, source, "CatBoost__Raw_plus_CCS", "Raw+CCS CatBoost")
    predictions = {
        "Ensemble | Raw HGB + Raw+CCS CatBoost": 0.5 * raw_hgb + 0.5 * augmented_catboost,
        "Ensemble | Raw+CCS HGB + Raw+CCS CatBoost": 0.5 * augmented_hgb + 0.5 * augmented_catboost,
    }
    if args.meps_oof is not None:
        meps_source = pd.read_csv(args.meps_oof, usecols=KEYS + [
            "all_disease_meps_teacher_catboost", "ccs_hgb_catboost_raw_plus_ccs_50",
            "all_disease_meps_teacher_hgb50",
        ])
        meps_catboost = align(ref, meps_source, "all_disease_meps_teacher_catboost",
                              "All-disease MEPS teacher CatBoost")
        saved_baseline = align(ref, meps_source, "ccs_hgb_catboost_raw_plus_ccs_50",
                               "Saved all-disease MEPS baseline")
        if not np.allclose(saved_baseline, predictions["Ensemble | Raw+CCS HGB + Raw+CCS CatBoost"],
                           rtol=0, atol=1e-12):
            raise ValueError("All-disease MEPS experiment baseline differs from controlled OOF")
        predictions.update({
            "Ensemble | Raw HGB + Raw+CCS CatBoost + MEPS":
                0.5 * raw_hgb + 0.5 * meps_catboost,
            "Ensemble | Raw+CCS HGB + Raw+CCS CatBoost + MEPS":
                0.5 * augmented_hgb + 0.5 * meps_catboost,
        })
        saved_meps_blend = align(ref, meps_source, "all_disease_meps_teacher_hgb50",
                                 "Saved all-disease MEPS blend")
        if not np.allclose(saved_meps_blend,
                           predictions["Ensemble | Raw+CCS HGB + Raw+CCS CatBoost + MEPS"],
                           rtol=0, atol=1e-12):
            raise ValueError("Reconstructed MEPS blend differs from original OOF")

    from diabetes_project.paper_comparison import top_fraction_metrics

    labels = ref.y_true.to_numpy(dtype=int)
    fold_ids = ref.Fold.to_numpy()
    rows = []
    for name, scores in predictions.items():
        for fold in range(1, 6):
            selected = fold_ids == fold
            rows.append({"Model": name.split(" | ", 1)[1], "Fold": fold,
                         **top_fraction_metrics(labels[selected], scores[selected])})
    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby("Model", as_index=False).agg(
        Folds=("Fold", "nunique"), PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"), AUROC_mean=("AUROC", "mean"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        F1_mean=("F1_score", "mean"), Brier_mean=("Brier_Score", "mean"),
    )
    summary["Pooled_OOF_PR_AUC"] = [
        average_precision_score(labels, predictions[f"Ensemble | {name}"])
        for name in summary.Model
    ]
    new = "Raw+CCS HGB + Raw+CCS CatBoost"
    control = "Raw HGB + Raw+CCS CatBoost"
    by_model = summary.set_index("Model")
    pairs = [(new, control)]
    if args.meps_oof is not None:
        pairs.extend([
            ("Raw HGB + Raw+CCS CatBoost + MEPS", control),
            ("Raw+CCS HGB + Raw+CCS CatBoost + MEPS", new),
            ("Raw+CCS HGB + Raw+CCS CatBoost + MEPS",
             "Raw HGB + Raw+CCS CatBoost + MEPS"),
        ])
    comparison = pd.DataFrame([{
        "Model": "Ensemble", "Treatment": treatment, "Control": comparator,
        "Delta_PR_AUC": by_model.loc[treatment, "PR_AUC_mean"] - by_model.loc[comparator, "PR_AUC_mean"],
        "Delta_AUROC": by_model.loc[treatment, "AUROC_mean"] - by_model.loc[comparator, "AUROC_mean"],
        "Delta_F1": by_model.loc[treatment, "F1_mean"] - by_model.loc[comparator, "F1_mean"],
        "Delta_Brier": by_model.loc[treatment, "Brier_mean"] - by_model.loc[comparator, "Brier_mean"],
    } for treatment, comparator in pairs])
    bootstrap = bootstrap_metrics(ref, predictions, comparison, args.bootstrap, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    comparison.to_csv(args.outdir / "comparison.csv", index=False)
    bootstrap.to_csv(args.outdir / "paired_bootstrap.csv", index=False)
    oof = ref.copy()
    for name, scores in predictions.items():
        oof[name.removeprefix("Ensemble | ").lower().replace("+", "_plus_").replace(" ", "_")] = scores
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "outer_folds": 5, "rows": len(ref),
        "patients": int(ref.patient_nbr.nunique()), "source_oof": str(args.oof.resolve()),
        "reference_oof": str(args.reference_oof.resolve()),
        "meps_oof": str(args.meps_oof.resolve()) if args.meps_oof is not None else None,
        "predictions": {name: "0.5 * " + ("HGB__Raw_plus_CCS" if name.startswith("Ensemble | Raw+CCS HGB")
                                    else "HGB__Raw") + " + 0.5 * "
                        + ("all_disease_meps_teacher_catboost" if name.endswith("+ MEPS")
                           else "CatBoost__Raw_plus_CCS") for name in predictions},
        "threshold_policy": "top 10% highest risk within each outer fold",
        "bootstrap": f"{args.bootstrap} paired patient resamples within fold",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("\nNew - Control, patient bootstrap\n" + bootstrap.to_string(index=False), flush=True)
    print(f"Saved to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/oof_all.csv.gz")
    parser.add_argument("--reference-oof", type=Path,
                        default=ROOT / "results/meps_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--meps-oof", type=Path, default=None,
                        help="Saved all-disease MEPS teacher CatBoost OOF; enables four-model comparison")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "results/raw_ccs_hgb_catboost_ensemble_seed777")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.seed != 777 or args.bootstrap < 1:
        parser.error("This comparison uses the saved seed-777 folds and positive bootstrap count")
    run(args)


if __name__ == "__main__":
    main()
