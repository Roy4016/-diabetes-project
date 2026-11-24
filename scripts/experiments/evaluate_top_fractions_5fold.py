#!/usr/bin/env python3
"""Re-evaluate audited seed-777 five-fold OOF at fixed Top-K alert budgets.

Each fold alerts its highest-risk K encounters, where K=floor(fraction * fold
size). The cutoff uses predicted risk only. No model is refit and no test
labels are used to choose a cutoff. PR-AUC/AUROC/Brier stay fixed across K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate_validation_f1_threshold import ROOT, KEYS, load_outer_scores
from diabetes_project.paper_comparison import top_fraction_metrics

METRIC_MAP = {"PR_AUC": "PR-AUC", "AUROC": "AUROC", "Precision": "Precision",
              "Recall": "Recall", "F1_score": "F1", "Brier_Score": "Brier",
              "Alerted": "Alerted", "True_Positive": "True_Positive",
              "Positive_Rate": "Positive_Rate"}
SUMMARY_METRICS = ["PR-AUC", "AUROC", "Precision", "Recall", "F1", "Brier"]


def evaluate(scores: pd.DataFrame, fractions: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [column for column in scores if column not in KEYS]
    if len(models) != len(set(models)) or scores.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("OOF models or patient-disjoint fold assignment invalid")
    fold_rows = []
    for fold, part in scores.groupby("Fold", sort=True):
        y = part.y_true.to_numpy(dtype=int)
        for model in models:
            probability = part[model].to_numpy(dtype=float)
            for fraction in fractions:
                measured = top_fraction_metrics(y, probability, fraction)
                fold_rows.append({"Model": model, "Fold": int(fold), "Top_Percent": 100 * fraction,
                                  "N": len(part), "Patients": int(part.patient_nbr.nunique()),
                                  **{METRIC_MAP[key]: value for key, value in measured.items()}})
    folds = pd.DataFrame(fold_rows)
    summary = (folds.groupby(["Model", "Top_Percent"], as_index=False)
               .agg(**{metric: (metric, "mean") for metric in SUMMARY_METRICS},
                    Folds=("Fold", "nunique"), Alerted_Mean=("Alerted", "mean"),
                    True_Positive_Mean=("True_Positive", "mean")))
    if not summary.Folds.eq(5).all():
        raise ValueError("Expected five outer folds for every model and budget")
    summary["Pooled_OOF_AP"] = summary.Model.map({
        name: average_precision_score(scores.y_true, scores[name]) for name in models})
    return folds, summary


def attach_extra_oof(scores: pd.DataFrame, path: Path) -> list[str]:
    extra = pd.read_csv(path)
    if len(extra) != len(scores) or extra.encounter_id.duplicated().any():
        raise ValueError(f"Extra OOF has a different cohort: {path}")
    extra = extra.set_index("encounter_id").loc[scores.encounter_id].reset_index()
    if not extra[KEYS].equals(scores[KEYS]):
        raise ValueError(f"Extra OOF encounter, patient, label or fold mismatch: {path}")
    aliases = {"ccs_only_hgb_catboost_raw_50": "ccs_only_catboost_raw_50"}
    added = []
    for original in extra.columns.difference(KEYS):
        name = aliases.get(original, original)
        probability = extra[original].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid extra OOF probability: {original}")
        if name in scores:
            if not np.allclose(scores[name].to_numpy(dtype=float), probability, atol=1e-12, rtol=0):
                raise ValueError(f"Extra OOF baseline differs from audited source: {original}")
        else:
            scores[name] = probability
            added.append(name)
    return added


def run(args):
    fractions = [value / 100 for value in args.top_percent]
    if any(not 0 < value < 1 for value in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError("--top-percent values must be unique and between 0 and 100")
    scores, audit = load_outer_scores(args)
    added = []
    for path in args.extra_oof:
        added.extend(attach_extra_oof(scores, path))
    folds, summary = evaluate(scores, fractions)
    args.outdir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    if 10 in args.top_percent:
        baseline = summary.loc[summary.Top_Percent.eq(10), ["Model", "Precision", "Recall", "F1"]].set_index("Model")
        comparison = summary.copy()
        for metric in ("Precision", "Recall", "F1"):
            comparison[f"Delta_{metric}_vs_Top10"] = comparison[metric] - comparison.Model.map(baseline[metric])
        comparison.to_csv(args.outdir / "comparison_vs_top10.csv", index=False)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": 777,
        "cohort": "99,343 encounters, 69,990 patients, fixed patient-disjoint five folds",
        "top_percent": args.top_percent,
        "alert_policy": "floor(fold_size * fraction) highest predicted risks; stable mergesort tie-break",
        "metrics": "arithmetic mean of five outer folds; AP=average_precision_score",
        "source_audit": audit,
        "ablation_oof": str(args.ablation_oof),
        "extra_oof": [str(path) for path in args.extra_oof],
        "added_models": added,
    }, indent=2) + "\n")
    print(summary[["Model", "Top_Percent", *SUMMARY_METRICS, "Alerted_Mean"]]
          .round(6).to_string(index=False))
    print(f"Saved to {args.outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-percent", nargs="+", type=float, default=[5, 10, 15, 20, 25])
    parser.add_argument("--ablation-oof", type=Path,
                        default=ROOT / "results/final_presentation_completion/catboost_feature_ablation_oof.csv")
    parser.add_argument("--extra-oof", nargs="*", type=Path, default=[],
                        help="Additional five-fold OOF; encounter/patient/label/fold and shared baselines are checked")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "results/top_fractions_5fold_seed777")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
