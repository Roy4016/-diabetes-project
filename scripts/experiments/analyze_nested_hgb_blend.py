#!/usr/bin/env python3
"""Fold-wise AP and patient-cluster bootstrap for nested HGB blend OOF output."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


DEFAULT_MODELS = [
    "hgb_tuned",
    "xgb_meps_tuned",
    "ensemble_fixed_50",
    "ensemble_nested",
]
DEFAULT_COMPARISONS = [
    ("ensemble_fixed_50", "hgb_tuned"),
    ("ensemble_nested", "hgb_tuned"),
    ("ensemble_nested", "ensemble_fixed_50"),
    ("xgb_meps_tuned", "hgb_tuned"),
]


def fold_metrics(oof, models):
    rows = []
    for fold, part in oof.groupby("Fold", sort=True):
        y = part["y_true"].to_numpy()
        for model in models:
            rows.append({
                "Fold": int(fold),
                "Experiment": model,
                "N_encounters": len(part),
                "N_patients": part["patient_nbr"].nunique(),
                "Prevalence": float(y.mean()),
                "PR_AUC_AP": float(average_precision_score(y, part[f"p_{model}"].to_numpy())),
            })
    return pd.DataFrame(rows)


def cluster_bootstrap(oof, models, comparisons, repetitions, seed):
    """Resample patients within each fold; each replicate averages fold APs equally."""
    rng = np.random.default_rng(seed)
    folds = []
    for fold, part in oof.groupby("Fold", sort=True):
        patient_ids, inverse = np.unique(part.patient_nbr.to_numpy(), return_inverse=True)
        folds.append((int(fold), part, len(patient_ids), inverse))

    # Cache bootstrap AP replicates for each model and fold to keep paired
    # comparisons on identical resamples.
    samples = {model: np.empty((repetitions, len(folds)), dtype=float) for model in models}
    for fold_index, (_fold, part, n_patients, inverse) in enumerate(folds):
        y = part.y_true.to_numpy()
        predictions = {model: part[f"p_{model}"].to_numpy() for model in models}
        for replicate in range(repetitions):
            counts = np.bincount(
                rng.integers(0, n_patients, size=n_patients), minlength=n_patients,
            )
            row_weights = counts[inverse]
            if row_weights.sum() == 0:
                raise RuntimeError("Patient bootstrap produced an empty fold")
            for model in models:
                samples[model][replicate, fold_index] = average_precision_score(
                    y, predictions[model], sample_weight=row_weights,
                )

    rows = []
    for left, right in comparisons:
        deltas = (samples[left] - samples[right]).mean(axis=1)
        observed = float(np.mean([
            average_precision_score(part.y_true, part[f"p_{left}"])
            - average_precision_score(part.y_true, part[f"p_{right}"])
            for _, part, _, _ in folds
        ]))
        rows.append({
            "Comparison": f"{left} - {right}",
            "MeanFold_Delta_PR_AUC": observed,
            "CI95_Lower": float(np.quantile(deltas, .025)),
            "CI95_Upper": float(np.quantile(deltas, .975)),
            "FractionPositive": float(np.mean(deltas > 0)),
            "BootstrapReplicates": repetitions,
            "ResamplingUnit": "patient within fold",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory containing oof_all.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")

    results_dir = Path(args.results_dir)
    input_path = results_dir / "oof_all.csv"
    if not input_path.is_file():
        parser.error(f"OOF file not found: {input_path}")
    oof = pd.read_csv(input_path, low_memory=False)
    required = {"Fold", "patient_nbr", "y_true"} | {f"p_{name}" for name in DEFAULT_MODELS}
    missing = sorted(required - set(oof.columns))
    if missing:
        parser.error(f"OOF file is missing columns: {missing}")
    if oof[list(required)].isna().any().any():
        parser.error("OOF file contains missing fold, label, patient, or prediction values")
    if not set(oof.y_true.unique()).issubset({0, 1}):
        parser.error("y_true must be binary 0/1")
    if (oof.groupby("patient_nbr").Fold.nunique() != 1).any():
        parser.error("A patient appears in multiple outer folds; patient-disjoint split was violated")
    for fold, part in oof.groupby("Fold"):
        if part.y_true.nunique() != 2:
            parser.error(f"Fold {fold} has only one outcome class")

    metrics = fold_metrics(oof, DEFAULT_MODELS)
    summary = metrics.groupby("Experiment").agg(
        PR_AUC_mean=("PR_AUC_AP", "mean"),
        PR_AUC_std=("PR_AUC_AP", "std"),
        Folds=("Fold", "nunique"),
    ).reset_index().sort_values("PR_AUC_mean", ascending=False)
    bootstrap = cluster_bootstrap(
        oof, DEFAULT_MODELS, DEFAULT_COMPARISONS, args.bootstrap, args.seed,
    )

    metrics.to_csv(results_dir / "fold_level_metrics.csv", index=False)
    summary.to_csv(results_dir / "fold_mean_summary.csv", index=False)
    bootstrap.to_csv(results_dir / "fold_stratified_patient_bootstrap.csv", index=False)
    print("Fold-level PR-AUC")
    print(metrics.pivot(index="Fold", columns="Experiment", values="PR_AUC_AP").round(6).to_string())
    print("\nMean across folds")
    print(summary.round(6).to_string(index=False))
    print("\nFold-stratified patient bootstrap")
    print(bootstrap.round(6).to_string(index=False))
    print(f"\nSaved analysis files to: {results_dir}")


if __name__ == "__main__":
    main()
