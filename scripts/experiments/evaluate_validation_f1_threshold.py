#!/usr/bin/env python3
"""Evaluate saved five-fold OOF with thresholds chosen on inner validation only.

Validation input is long-form: Model, OuterFold, encounter_id, patient_nbr,
y_true, probability. Each row must be an outer-training encounter. This program
never uses outer-test labels to select thresholds. Unavailable models are listed
in missing_models.csv rather than assigned an invented F1 score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)

from complete_final_presentation import KEYS, MODEL_SPECS, ROOT, load_sources


def best_f1_threshold(y, probability):
    """Maximize validation F1; ties use the highest (most selective) threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    values = np.divide(2 * precision[:-1] * recall[:-1],
                       precision[:-1] + recall[:-1],
                       out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) > 0)
    maximum = values.max()
    winner = np.flatnonzero(np.isclose(values, maximum, atol=1e-12, rtol=0))[-1]
    return float(thresholds[winner]), float(maximum)


def load_outer_scores(args):
    sources, reference, audit = load_sources()
    scores = reference.copy()
    for name, source, column, _ in MODEL_SPECS:
        scores[name] = sources[source][column].to_numpy(dtype=float)
    scores["raw_hgb_catboost_raw_50"] = (scores.roy_hgb_46 + scores.catboost_raw) / 2
    ablation = pd.read_csv(args.ablation_oof)
    aligned = ablation.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not aligned[KEYS].equals(reference):
        raise ValueError("CatBoost ablation OOF keys differ from reference")
    for name in ("E1_no_number_inpatient", "E2_no_raw_diagnoses", "E3_no_all_diagnosis_derived"):
        if name in aligned:
            scores[name] = aligned[name].to_numpy(dtype=float)
    for name in scores.columns.difference(KEYS):
        p = scores[name].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid outer probability for {name}")
    return scores, audit


def validate_inner(inner, outer):
    required = {"Model", "OuterFold", "encounter_id", "patient_nbr", "y_true", "probability"}
    if missing := required - set(inner):
        raise ValueError(f"Inner predictions lack columns: {sorted(missing)}")
    if inner.duplicated(["Model", "OuterFold", "encounter_id"]).any():
        raise ValueError("Duplicate model/fold/encounter in inner predictions")
    if not inner.OuterFold.isin(range(1, 6)).all():
        raise ValueError("OuterFold must be 1..5")
    p = pd.to_numeric(inner.probability, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Inner predictions must be probabilities in [0,1]")
    reference = outer[KEYS].rename(columns={"Fold": "ActualFold", "patient_nbr": "ActualPatient",
                                                "y_true": "ActualLabel"})
    checked = inner.merge(reference, on="encounter_id", how="left", validate="many_to_one")
    if checked.ActualFold.isna().any() or not checked.patient_nbr.eq(checked.ActualPatient).all() or not checked.y_true.eq(checked.ActualLabel).all():
        raise ValueError("Inner encounter, patient, or label differs from audited OOF")
    if checked.OuterFold.eq(checked.ActualFold).any():
        raise ValueError("Inner validation includes patients from its outer test fold")
    for (model, fold), part in checked.groupby(["Model", "OuterFold"]):
        if part.patient_nbr.nunique() < 100 or part.y_true.nunique() != 2:
            raise ValueError(f"Too few validation patients or only one class: {model}, fold {fold}")
    return checked


def run(args):
    outer, audit = load_outer_scores(args)
    parts = [pd.read_csv(path) for path in args.validation]
    inner = validate_inner(pd.concat(parts, ignore_index=True), outer)
    if args.assignments is not None:
        assignments = pd.read_csv(args.assignments)
        columns = KEYS + [f"inner_val_{i}" for i in range(1, 6)]
        assignments = assignments.set_index("encounter_id").loc[outer.encounter_id].reset_index()
        if not assignments[KEYS].equals(outer[KEYS]):
            raise ValueError("Saved inner-fold assignments differ from OOF")
        matched = inner.merge(assignments[["encounter_id", *columns[4:]]], on="encounter_id", validate="many_to_one")
        flags = np.array([matched.loc[matched.OuterFold.eq(i), f"inner_val_{i}"].eq(1).all()
                          for i in range(1, 6)])
        if not flags.all():
            raise ValueError("Validation predictions do not match saved inner_val_k assignments")
    args.outdir.mkdir(parents=True, exist_ok=True)
    records, chosen = [], []
    for model in outer.columns.difference(KEYS):
        if model not in set(inner.Model):
            continue
        for fold in range(1, 6):
            val = inner.loc[inner.Model.eq(model) & inner.OuterFold.eq(fold)]
            if val.empty:
                continue
            threshold, validation_f1 = best_f1_threshold(val.y_true.to_numpy(dtype=int), val.probability.to_numpy(dtype=float))
            test = outer.loc[outer.Fold.eq(fold)]
            y = test.y_true.to_numpy(dtype=int)
            p = test[model].to_numpy(dtype=float)
            predicted = p >= threshold
            precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
            records.append({"Model": model, "Fold": fold, "N": len(test), "Validation_N": len(val),
                            "Threshold": threshold, "Validation_F1": validation_f1,
                            "Alerted": int(predicted.sum()), "Alert_Fraction": float(predicted.mean()),
                            "PR-AUC": float(average_precision_score(y, p)),
                            "AUROC": float(roc_auc_score(y, p)), "Precision": float(precision),
                            "Recall": float(recall), "F1": float(f1),
                            "Brier": float(brier_score_loss(y, p))})
            chosen.append({"Model": model, "Fold": fold, "Threshold": threshold,
                           "Validation_F1": validation_f1, "Validation_N": len(val)})
    folds = pd.DataFrame(records)
    if folds.empty:
        raise ValueError("No matching model/fold validation predictions")
    complete = folds.groupby("Model").Fold.nunique()
    summary = folds.loc[folds.Model.isin(complete[complete.eq(5)].index)].groupby("Model", as_index=False).agg({
        "PR-AUC": "mean", "AUROC": "mean", "Precision": "mean", "Recall": "mean",
        "F1": "mean", "Brier": "mean", "Alert_Fraction": "mean"})
    summary["Pooled_OOF_AP"] = [average_precision_score(outer.y_true, outer[model]) for model in summary.Model]
    missing = pd.DataFrame([{"Model": model, "Available_Folds": int(complete.get(model, 0)),
                             "Reason": "Missing audited inner-validation probabilities for one or more folds"}
                            for model in outer.columns.difference(KEYS) if complete.get(model, 0) != 5])
    folds.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    pd.DataFrame(chosen).to_csv(args.outdir / "thresholds.csv", index=False)
    missing.to_csv(args.outdir / "missing_models.csv", index=False)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": 777, "outer_split": "fixed patient-disjoint five-fold, 99343 rows",
        "inner_assignments": str(args.assignments), "validation_files": [str(p) for p in args.validation],
        "policy": "max validation F1, highest threshold on tie; apply unchanged to outer OOF",
        "metric_aggregation": "arithmetic mean of five outer fold values",
        "source_audit": audit,
    }, indent=2) + "\n")
    print(summary.round(6).to_string(index=False))
    print(f"Missing models: {len(missing)}; see {args.outdir / 'missing_models.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--assignments", type=Path, default=ROOT.parent / "Bhuvan_Rajpoot_SSRN/5-fold-reproduce/results/five_fold/fold_assignments.csv.gz")
    parser.add_argument("--ablation-oof", type=Path, default=ROOT / "results/final_presentation_completion/catboost_feature_ablation_oof.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/validation_f1_threshold_seed777")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
