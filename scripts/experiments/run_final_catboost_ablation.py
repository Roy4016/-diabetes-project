#!/usr/bin/env python3
"""Retrain Raw ICD-9 CatBoost ablations on the audited seed-777 UCI folds."""

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

from diabetes_project.paper_comparison import load_comparison_cohort, make_paper_catboost_features  # noqa: E402
from run_cms_group_prior_5fold import checksum  # noqa: E402
from run_paper_fair_comparison import choose_catboost, fit_catboost  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
DROP = {"E1_no_number_inpatient": ["number_inpatient"],
        "E2_no_raw_diagnoses": ["diag_1", "diag_2", "diag_3"],
        "E3_no_all_diagnosis_derived": ["diag_1", "diag_2", "diag_3", "number_diagnoses"]}


def run(args) -> None:
    if args.seed != 777 or not 1 <= args.max_folds <= 5:
        raise ValueError("This presentation comparison requires seed 777 and 1..5 folds")
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError("Duplicate ablation names")
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_oof, usecols=KEYS + ["catboost_raw"])
    if len(reference) != 99343 or reference.encounter_id.duplicated().any() or set(reference.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Reference is not the expected complete five-fold OOF")
    if reference.patient_nbr.nunique() != 69990 or reference.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Patient count or patient-disjoint split mismatch")
    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(reference) or cohort.encounter_id.duplicated().any():
        raise ValueError("UCI cohort differs from reference")
    cohort = cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), reference.patient_nbr.to_numpy()) or not np.array_equal(
        cohort.target.to_numpy(), reference.y_true.to_numpy()
    ):
        raise ValueError("UCI patient or label mismatch")
    x, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    if len(x.columns) != 46 or any(c.startswith(("ccs_", "diag_family", "diagnosis_")) for c in x.columns):
        raise ValueError("Raw CatBoost should contain exactly 46 predictors and no diagnosis-derived columns")
    print("[Audit] exact 99,343-row, 69,990-patient seed-777 split; 46 raw predictors; E3 additionally removes number_diagnoses", flush=True)

    provenance = {
        "reference_oof_sha256": checksum(args.reference_oof),
        "hospital_sha256": checksum(args.hospital),
        "seed": args.seed, "alert_fraction": .10,
        "model": "CatBoostClassifier CPU; original choose_catboost + fit_catboost functions",
        "inner_split": "GroupShuffleSplit 20%, random_state=seed+fold, outer-train patients only",
        "grid": {"depths": args.depths, "iterations": args.iterations,
                 "learning_rate": .035, "l2_leaf_reg": {"4": 5., "6": 15.}, "model_seed": 2026},
        "ablations": DROP,
        "E3_status": "distinct from E2: additionally drops number_diagnoses, the diagnosis-count feature",
    }
    metadata_path = args.outdir / "catboost_ablation_run.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous != provenance:
            legacy = dict(previous)
            legacy.pop("E3_status", None)
            legacy["ablations"] = {key: value for key, value in legacy["ablations"].items()
                                   if key != "E3_no_all_diagnosis_derived"}
            expected_legacy = dict(provenance)
            expected_legacy.pop("E3_status", None)
            expected_legacy["ablations"] = {key: value for key, value in expected_legacy["ablations"].items()
                                            if key != "E3_no_all_diagnosis_derived"}
            if legacy != expected_legacy:
                raise ValueError("Saved ablation metadata differs; use a new output directory")
    metadata_path.write_text(json.dumps(provenance, indent=2) + "\n")
    folds = sorted(reference.Fold.unique())[:args.max_folds]
    trial_rows = []
    for variant in args.ablations:
        columns = x.columns.drop(DROP[variant]).tolist()
        categories = [name for name in categorical if name in columns]
        for fold in folds:
            target = args.outdir / f"{variant}_fold_{fold}.csv.gz"
            if target.exists():
                saved = pd.read_csv(target)
                expected = reference.loc[reference.Fold.eq(fold), KEYS].reset_index(drop=True)
                if not saved[KEYS].equals(expected) or saved[variant].isna().any():
                    raise ValueError(f"Saved partial fold does not match the reference: {target}")
                print(f"[Fold {fold}] {variant}: verified saved predictions", flush=True)
                continue
            train = reference.Fold.ne(fold).to_numpy()
            test = ~train
            train_x = x.loc[train, columns].reset_index(drop=True)
            test_x = x.loc[test, columns].reset_index(drop=True)
            y = reference.loc[train, "y_true"].to_numpy(dtype=int)
            patients = reference.loc[train, "patient_nbr"].to_numpy()
            chosen, trials = choose_catboost(train_x, y, patients, categories, int(fold), args)
            for trial in trials:
                trial_rows.append({"Ablation": variant, **trial})
            model, score = fit_catboost(train_x, y, test_x, categories, chosen, args)
            part = reference.loc[test, KEYS].reset_index(drop=True)
            part[variant] = score
            part.to_csv(target, index=False)
            print(f"[Fold {fold}] {variant}: depth={chosen['Depth']} trees={chosen['Trees']} "
                  f"AP={average_precision_score(part.y_true, score):.6f}", flush=True)
            (args.outdir / f"{variant}_fold_{fold}_selection.json").write_text(
                json.dumps(chosen, indent=2) + "\n"
            )
    if trial_rows:
        pd.DataFrame(trial_rows).to_csv(args.outdir / "catboost_ablation_inner_trials_current_run.csv", index=False)
    complete_variants = [variant for variant in DROP if all(
        (args.outdir / f"{variant}_fold_{fold}.csv.gz").exists() for fold in range(1, 6))]
    if args.max_folds == 5 and complete_variants:
        joined = reference[KEYS + ["catboost_raw"]].copy()
        for variant in complete_variants:
            pieces = pd.concat([pd.read_csv(args.outdir / f"{variant}_fold_{fold}.csv.gz")
                                for fold in folds], ignore_index=True)
            if len(pieces) != len(joined) or pieces.encounter_id.duplicated().any():
                raise ValueError("Ablation OOF incomplete")
            pieces = pieces.set_index("encounter_id").loc[joined.encounter_id].reset_index()
            if not pieces[KEYS].equals(joined[KEYS]):
                raise ValueError("Ablation OOF differs from the reference split")
            joined[variant] = pieces[variant].to_numpy()
        joined.to_csv(args.outdir / "catboost_feature_ablation_oof.csv", index=False)
        print(f"Saved full ablation OOF: {args.outdir / 'catboost_feature_ablation_oof.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/final_presentation_completion")
    parser.add_argument("--ablations", nargs="+", choices=sorted(DROP), default=list(DROP))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--iterations", type=int, nargs="+", default=[250, 500, 750])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
