#!/usr/bin/env python3
"""Controlled raw, CCS replacement and raw+CCS HGB on saved UCI folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import encode_frames, choose_hgb, hgb_model
from run_xgb_5fold_recheck import read_split
from train_xgboost_ahrq_ccs_prefix import load_ccs

from diabetes_project.paper_comparison import top_fraction_metrics

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"raw": "hgb_raw", "replace": "hgb_ccs_replace", "append": "hgb_raw_plus_ccs"}


def run(args):
    cohort, fold_ids = read_split(args.input, args.reference_oof)
    raw_columns = raw_discharge_predictors(cohort)
    mapping, _ = load_ccs(args.ccs_file)
    frame = attach_uci_ccs(cohort, mapping)
    columns = {
        "raw": raw_columns,
        "replace": [f"ccs_{column}" if column in DIAG else column for column in raw_columns],
        "append": raw_columns + [f"ccs_{column}" for column in DIAG],
    }
    if [len(columns[name]) for name in NAMES] != [46, 46, 49]:
        raise ValueError("Unexpected raw/replace/append feature counts")
    labels = cohort.target.to_numpy(dtype=int)
    patients = cohort.patient_nbr.to_numpy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_rows, predictions, selections, counts = [], [], [], []
    for fold in range(1, 6):
        outer = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        train_rel, val_rel = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=args.seed + fold,
        ).split(frame.iloc[outer], labels[outer], patients[outer]))
        inner, val = outer[train_rel], outer[val_rel]
        if set(patients[inner]) & set(patients[val]) or set(patients[outer]) & set(patients[test]):
            raise ValueError("Patient overlap")
        for variant, name in NAMES.items():
            feature_names = columns[variant]
            x_inner, x_val, _, inner_prep = encode_frames(
                frame.iloc[inner], frame.iloc[val], frame.iloc[val],
                feature_names, args.min_frequency, return_transformer=True,
            )
            val_ap, params, iterations, _ = choose_hgb(
                x_inner, x_val, labels[inner], labels[val], args.seed + fold,
            )
            x_outer, _, x_test, outer_prep = encode_frames(
                frame.iloc[outer], frame.iloc[outer], frame.iloc[test],
                feature_names, args.min_frequency, return_transformer=True,
            )
            final = hgb_model(args.seed + fold, {**params, "max_iter": iterations}, early_stopping=False)
            final.fit(x_outer, labels[outer])
            probability = final.predict_proba(x_test)[:, 1]
            metric = top_fraction_metrics(labels[test], probability)
            metrics_rows.append({"Model": name, "Fold": fold, **metric})
            selections.append({"Model": name, "Fold": fold, "Validation_AP": val_ap,
                               "Selected_Iterations": iterations, "Parameters": json.dumps(params)})
            counts.append({"Model": name, "Fold": fold, "Input_Features": len(feature_names),
                           "Inner_Encoded_Features": x_inner.shape[1],
                           "Outer_Encoded_Features": x_outer.shape[1]})
            predictions.append(pd.DataFrame({
                "encounter_id": cohort.encounter_id.iloc[test].to_numpy(),
                "patient_nbr": patients[test], "y_true": labels[test], "Fold": fold,
                "Model": name, "probability": probability,
            }))
            print(f"[Fold {fold}] {name} AP={metric['PR_AUC']:.6f} AUROC={metric['AUROC']:.6f}", flush=True)
        pd.DataFrame(metrics_rows).to_csv(args.outdir / "fold_metrics.csv", index=False)
        pd.DataFrame(selections).to_csv(args.outdir / "selections.csv", index=False)
        pd.DataFrame(counts).to_csv(args.outdir / "feature_counts.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(args.outdir / "oof_all.csv.gz", index=False)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "folds": 5,
        "reference_oof": str(args.reference_oof.resolve()),
        "input": str(args.input.resolve()), "ccs_file": str(args.ccs_file.resolve()),
        "features": columns, "min_frequency": args.min_frequency,
        "selection": "patient-disjoint inner 20% validation; five HGB candidates; max AP; refit outer train",
        "threshold_policy": "top 10% highest risk in each outer test fold",
        "class_weight": None, "external_data": None,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--reference-oof", type=Path,
                        default=ROOT / "results/meps_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--ccs-file", type=Path,
                        default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/hgb_controlled")
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.seed != 777:
        parser.error("The reference OOF fixes outer folds to seed 777")
    run(args)


if __name__ == "__main__":
    main()
