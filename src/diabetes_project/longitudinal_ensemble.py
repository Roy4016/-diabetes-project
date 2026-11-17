"""Combine independently trained longitudinal blend runs without test-set selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .external_embedding import _patient_bootstrap
from .longitudinal_meps import _paired_patient_bootstrap


KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
PROBABILITIES = [
    "p_clinical", "p_risk_mtl", "p_blend_mtl",
    "p_risk_shuffled_mtl", "p_blend_shuffled_mtl",
]


def run_ensemble(input_dirs, output_dir, seed=777, bootstrap_count=1000):
    if len(input_dirs) < 2:
        raise ValueError("At least two independent runs are required")
    frames, seeds = [], []
    reference = None
    for directory in input_dirs:
        config_path = Path(directory) / "config.json"
        if config_path.exists():
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            seeds.append(metadata.get("config", {}).get("seed"))
        frame = pd.read_csv(Path(directory) / "oof_all.csv").sort_values(KEYS).reset_index(drop=True)
        missing = set(KEYS + PROBABILITIES).difference(frame.columns)
        if missing:
            raise ValueError(f"{directory} is missing columns: {sorted(missing)}")
        if reference is None:
            reference = frame[KEYS].copy()
        elif not reference.equals(frame[KEYS]):
            raise ValueError(f"OOF identities or labels differ in {directory}")
        frames.append(frame)

    combined = reference.copy()
    for column in PROBABILITIES:
        combined[column] = np.mean([frame[column].to_numpy() for frame in frames], axis=0)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "oof_all.csv", index=False)

    rows = []
    for fold, part in combined.groupby("Fold", sort=True):
        for column in PROBABILITIES:
            probability = part[column].to_numpy(); labels = part.y_true.to_numpy()
            rows.append({
                "Fold": int(fold), "Experiment": column.removeprefix("p_"),
                "AP": float(average_precision_score(labels, probability)),
                "AUROC": float(roc_auc_score(labels, probability)),
                "Brier": float(brier_score_loss(labels, probability)),
            })
    fold_results = pd.DataFrame(rows); fold_results.to_csv(output / "fold_results.csv", index=False)
    summary = fold_results.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"),
        AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
        Brier_mean=("Brier", "mean"), Brier_std=("Brier", "std"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)

    variants = ("clinical", "risk_mtl", "blend_mtl", "risk_shuffled_mtl", "blend_shuffled_mtl")
    bootstrap = _patient_bootstrap(combined, variants, seed, bootstrap_count)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    external_signal = _paired_patient_bootstrap(combined, [
        ("risk_mtl", "risk_shuffled_mtl"), ("blend_mtl", "blend_shuffled_mtl"),
    ], seed, bootstrap_count)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    pooled = []
    for column in PROBABILITIES:
        probability = combined[column].to_numpy(); labels = combined.y_true.to_numpy()
        pooled.append({
            "Experiment": column.removeprefix("p_"),
            "AP": float(average_precision_score(labels, probability)),
            "AUROC": float(roc_auc_score(labels, probability)),
            "Brier": float(brier_score_loss(labels, probability)),
        })
    pooled = pd.DataFrame(pooled); pooled.to_csv(output / "pooled_metrics.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "input_dirs": [str(Path(value)) for value in input_dirs],
        "seeds": seeds, "bootstrap": bootstrap_count,
        "note": "Probabilities were averaged across all pre-specified seeds; no seed was selected by test performance.",
    }, indent=2), encoding="utf-8")
    return summary, pooled, bootstrap, external_signal
