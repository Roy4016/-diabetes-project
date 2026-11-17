#!/usr/bin/env python3
"""Combine HGB and XGBoost OOF predictions with a fixed 50/50 rule."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


KEYS = ["encounter_id", "patient_nbr", "y_true"]


def bootstrap(frame, comparisons, repetitions, seed):
    patients, inverse = np.unique(frame.patient_nbr.to_numpy(), return_inverse=True)
    labels = frame.y_true.to_numpy()
    rng = np.random.default_rng(seed + 19000)
    rows = []
    for left, right in comparisons:
        a, b = frame[f"p_{left}"].to_numpy(), frame[f"p_{right}"].to_numpy()
        observed = average_precision_score(labels, a) - average_precision_score(labels, b)
        deltas = []
        for _ in range(repetitions):
            counts = np.bincount(
                rng.integers(0, len(patients), size=len(patients)), minlength=len(patients),
            )
            weight = counts[inverse]
            deltas.append(
                average_precision_score(labels, a, sample_weight=weight)
                - average_precision_score(labels, b, sample_weight=weight)
            )
        rows.append({
            "Comparison": f"{left} - {right}", "Delta_PR_AUC": observed,
            "CI95_Lower": float(np.quantile(deltas, .025)),
            "CI95_Upper": float(np.quantile(deltas, .975)),
            "FractionPositive": float(np.mean(np.asarray(deltas) > 0)),
            "Bootstraps": repetitions,
        })
    return pd.DataFrame(rows)


def load_pair(hgb_dir, xgb_dir, seed):
    hgb = pd.read_csv(Path(hgb_dir) / "oof_all.csv")
    xgb = pd.read_csv(Path(xgb_dir) / "oof_all.csv")
    frame = hgb[KEYS + ["Fold", "p_Clinical_HGB"]].merge(
        xgb[KEYS + ["Fold", "p_Clinical", "p_Direct8_MEPS_Both8"]],
        on=KEYS, suffixes=("_hgb", "_xgb"), validate="one_to_one",
    )
    frame["p_hgb"] = frame.p_Clinical_HGB
    frame["p_clinical_xgb"] = frame.p_Clinical
    frame["p_meps_xgb"] = frame.p_Direct8_MEPS_Both8
    frame["p_no_external"] = (frame.p_hgb + frame.p_clinical_xgb) / 2
    frame["p_external"] = (frame.p_hgb + frame.p_meps_xgb) / 2
    frame["Seed"] = seed
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgb-dirs", nargs="+", required=True)
    parser.add_argument("--xgb-dirs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--outdir", default="results/meps_hgb_xgb_ensemble")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if not (len(args.hgb_dirs) == len(args.xgb_dirs) == len(args.seeds)):
        parser.error("--hgb-dirs, --xgb-dirs, and --seeds must have equal lengths")
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    frames = [load_pair(h, x, s) for h, x, s in zip(args.hgb_dirs, args.xgb_dirs, args.seeds)]
    variants = ["hgb", "clinical_xgb", "meps_xgb", "no_external", "external"]
    seed_rows = []
    for frame, seed in zip(frames, args.seeds):
        for variant in variants:
            fold_values = [
                average_precision_score(part.y_true, part[f"p_{variant}"])
                for _, part in frame.groupby("Fold_hgb")
            ]
            seed_rows.append({
                "Seed": seed, "Experiment": variant,
                "PR_AUC_mean": float(np.mean(fold_values)),
                "PR_AUC_std": float(np.std(fold_values, ddof=1)),
                "Pooled_PR_AUC": float(average_precision_score(frame.y_true, frame[f"p_{variant}"])),
            })
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(output / "seed_summary.csv", index=False)

    combined = None
    for frame, seed in zip(frames, args.seeds):
        current = frame[KEYS + [f"p_{name}" for name in variants]].rename(
            columns={f"p_{name}": f"p_{name}_{seed}" for name in variants},
        )
        combined = current if combined is None else combined.merge(
            current, on=KEYS, validate="one_to_one",
        )
    for name in variants:
        combined[f"p_{name}"] = combined[[f"p_{name}_{seed}" for seed in args.seeds]].mean(axis=1)
    combined.to_csv(output / "oof_all.csv", index=False)
    ensemble_summary = pd.DataFrame([{
        "Experiment": name,
        "Pooled_PR_AUC": average_precision_score(combined.y_true, combined[f"p_{name}"]),
    } for name in variants])
    ensemble_summary.to_csv(output / "summary.csv", index=False)
    comparisons = [
        ("external", "hgb"), ("external", "meps_xgb"),
        ("external", "no_external"),
    ]
    intervals = bootstrap(combined, comparisons, args.bootstrap, args.seeds[0])
    intervals.to_csv(output / "patient_bootstrap.csv", index=False)
    print("\nPer-seed summary\n", seed_summary.round(6).to_string(index=False))
    print("\nAveraged-OOF summary\n", ensemble_summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", intervals.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
