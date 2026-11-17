
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupShuffleSplit,
)

from run_cdc_xgb import (
    load_hospital,
    make_features,
)

from run_meps_multitask import add_clinical


def get_feature_columns(df, clinical):
    excluded = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target",
        *clinical,
    }

    original = [
        c for c in df.columns
        if (
            c not in excluded
            and not c.startswith("bridge_")
            and not c.startswith("meps_")
        )
    ]

    return original + clinical


def convert_matrix(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy()
        names = [str(x) for x in matrix.columns]
    else:
        values = np.asarray(matrix)
        names = [
            f"feature_{i:04d}"
            for i in range(values.shape[1])
        ]

    values = np.asarray(values, dtype=np.float32)

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return values, names


def process_split(train, test, features):
    # The original make_features() performs
    # train-fitted preprocessing on train/val/test.
    matrices = make_features(
        train,
        train,
        test,
        features,
    )

    x_train, names_train = convert_matrix(
        matrices[0]
    )

    x_test, names_test = convert_matrix(
        matrices[2]
    )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            "Train/Test feature dimensions mismatch"
        )

    if len(names_train) != len(names_test):
        raise ValueError(
            "Feature name lengths mismatch"
        )

    # Prefer names from fitted train matrix.
    return x_train, x_test, names_train


def save_dataset(
    path,
    raw,
    matrix,
    names,
):
    if len(raw) != len(matrix):
        raise ValueError(
            "Row count mismatch"
        )

    output = pd.DataFrame(
        matrix,
        columns=names,
    )

    if output.columns.duplicated().any():
        raise ValueError(
            "Duplicate output feature names"
        )

    output.insert(
        0,
        "target",
        raw["target"].to_numpy(dtype=np.int8),
    )

    output.insert(
        0,
        "patient_nbr",
        raw["patient_nbr"].to_numpy(),
    )

    output.insert(
        0,
        "encounter_id",
        raw["encounter_id"].to_numpy(),
    )

    output.to_csv(
        path,
        index=False,
        float_format="%.8g",
    )

    return output.shape


def assert_disjoint(a, b, context):
    overlap = (
        set(a["patient_nbr"])
        & set(b["patient_nbr"])
    )

    if overlap:
        raise ValueError(
            f"{context}: {len(overlap)} "
            "patients appear in both splits"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital",
        default="diabetic_data.csv",
    )

    parser.add_argument(
        "--outdir",
        default="processed_diabetes_5fold",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=777,
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2")

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("========== LOAD HOSPITAL ==========")

    hospital = load_hospital(
        args.hospital
    ).reset_index(drop=True)

    hospital, clinical = add_clinical(
        hospital
    )

    required = [
        "encounter_id",
        "patient_nbr",
        "target",
    ]

    for col in required:
        if col not in hospital:
            raise ValueError(
                f"Missing required column: {col}"
            )

    features = get_feature_columns(
        hospital,
        clinical,
    )

    print(
        f"Rows: {len(hospital):,}"
    )

    print(
        f"Unique patients: "
        f"{hospital.patient_nbr.nunique():,}"
    )

    print(
        f"Readmission prevalence: "
        f"{hospital.target.mean():.5f}"
    )

    print(
        f"Clinical features: {len(clinical)}"
    )

    print(
        f"Model source features: {len(features)}"
    )

    assignments = hospital[
        [
            "encounter_id",
            "patient_nbr",
            "target",
        ]
    ].copy()

    assignments.insert(
        0,
        "row_index",
        np.arange(len(hospital)),
    )

    fold_summary = []

    cv = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    splits = cv.split(
        hospital,
        hospital["target"],
        groups=hospital["patient_nbr"],
    )

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        print(
            f"\n========== FOLD {fold}/{args.folds} ==========",
            flush=True,
        )

        fold_dir = outdir / f"fold_{fold}"
        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train = hospital.iloc[
            train_idx
        ].copy()

        test = hospital.iloc[
            test_idx
        ].copy()

        assert_disjoint(
            train, test,
            f"Fold {fold}: outer split",
        )

        # Same inner split settings as the
        # original MEPS multitask experiment.
        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=args.seed + fold,
        )

        inner_local, val_local = next(
            inner_splitter.split(
                train,
                groups=train["patient_nbr"],
            )
        )

        inner_global = train_idx[inner_local]
        val_global = train_idx[val_local]

        inner_train = hospital.iloc[
            inner_global
        ].copy()

        validation = hospital.iloc[
            val_global
        ].copy()

        assert_disjoint(
            inner_train,
            validation,
            f"Fold {fold}: inner split",
        )

        assert_disjoint(
            inner_train,
            test,
            f"Fold {fold}: inner vs test",
        )

        assert_disjoint(
            validation,
            test,
            f"Fold {fold}: val vs test",
        )

        # Outer train/test preprocessing.
        # Fit on outer train only.
        x_train, x_test, outer_names = (
            process_split(
                train,
                test,
                features,
            )
        )

        # Inner train/validation preprocessing.
        # Fit on inner train only.
        x_inner, x_val, inner_names = (
            process_split(
                inner_train,
                validation,
                features,
            )
        )

        train_shape = save_dataset(
            fold_dir / "train.csv",
            train,
            x_train,
            outer_names,
        )

        test_shape = save_dataset(
            fold_dir / "test.csv",
            test,
            x_test,
            outer_names,
        )

        inner_shape = save_dataset(
            fold_dir / "inner_train.csv",
            inner_train,
            x_inner,
            inner_names,
        )

        val_shape = save_dataset(
            fold_dir / "val.csv",
            validation,
            x_val,
            inner_names,
        )

        # Save exact indices of every split.
        np.savez_compressed(
            fold_dir / "indices.npz",
            train_idx=train_idx,
            test_idx=test_idx,
            inner_train_idx=inner_global,
            val_idx=val_global,
        )

        (fold_dir / "outer_features.json").write_text(
            json.dumps(
                outer_names,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (fold_dir / "inner_features.json").write_text(
            json.dumps(
                inner_names,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Record each patient's encounter assignment.
        assignment = np.full(
            len(hospital),
            "train",
            dtype=object,
        )

        assignment[test_idx] = "test"
        assignment[inner_global] = "inner_train"
        assignment[val_global] = "val"

        assignments[f"fold_{fold}_split"] = assignment

        info = {
            "fold": fold,
            "seed": args.seed,
            "train_rows": len(train),
            "test_rows": len(test),
            "inner_train_rows": len(inner_train),
            "val_rows": len(validation),
            "train_patients": int(
                train.patient_nbr.nunique()
            ),
            "test_patients": int(
                test.patient_nbr.nunique()
            ),
            "train_prevalence": float(
                train.target.mean()
            ),
            "test_prevalence": float(
                test.target.mean()
            ),
            "outer_feature_count": int(
                x_train.shape[1]
            ),
            "inner_feature_count": int(
                x_inner.shape[1]
            ),
            "patient_overlap": 0,
        }

        fold_summary.append(info)

        (fold_dir / "info.json").write_text(
            json.dumps(
                info,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Train: {train_shape}, "
            f"Test: {test_shape}",
            flush=True,
        )

        print(
            f"Inner train: {inner_shape}, "
            f"Validation: {val_shape}",
            flush=True,
        )

        print(
            "Patient overlap: 0",
            flush=True,
        )

    assignments.to_csv(
        outdir / "fold_assignments.csv",
        index=False,
    )

    summary = pd.DataFrame(
        fold_summary
    )

    summary.to_csv(
        outdir / "fold_summary.csv",
        index=False,
    )

    metadata = {
        "dataset": "UCI Diabetes 130-US Hospitals",
        "hospital_source": args.hospital,
        "task": "30-day readmission prediction",
        "label": "target",
        "n_rows": len(hospital),
        "n_patients": int(
            hospital.patient_nbr.nunique()
        ),
        "n_folds": args.folds,
        "seed": args.seed,
        "outer_split": "StratifiedGroupKFold",
        "outer_group": "patient_nbr",
        "inner_split": "GroupShuffleSplit",
        "inner_test_size": 0.20,
        "inner_seed_rule": "seed + fold",
        "clinical_feature_count": len(clinical),
        "source_feature_columns": features,
        "clinical_feature_columns": clinical,
        "note": (
            "The preprocessing for train/test is "
            "fitted on the outer training fold. "
            "The preprocessing for inner_train/val "
            "is fitted on the inner training split. "
            "Each fold uses independent preprocessing."
        ),
    }

    (outdir / "dataset_info.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# Processed UCI Diabetes 5-Fold Dataset

## Dataset
UCI Diabetes 130-US Hospitals.

Rows: {len(hospital):,}
Unique patients: {hospital.patient_nbr.nunique():,}

## Target
target = 1: readmitted within 30 days
target = 0: not readmitted within 30 days

## Splitting
Outer split:
StratifiedGroupKFold(n_splits={args.folds},
shuffle=True, random_state={args.seed})

Group variable: patient_nbr

Inner validation:
GroupShuffleSplit(test_size=0.20,
random_state={args.seed} + fold_number)

All records from the same patient remain
in the same partition for any given fold.

## Files
fold_assignments.csv:
Original encounter ID, patient ID, target,
and partition assignment for every fold.

fold_summary.csv:
Sizes, patient counts, and prevalence.

fold_N/train.csv:
Preprocessed outer training set.

fold_N/test.csv:
Preprocessed outer test set.

fold_N/inner_train.csv:
Preprocessed inner training set.

fold_N/val.csv:
Preprocessed inner validation set.

fold_N/indices.npz:
Exact original row indices for all partitions.

fold_N/outer_features.json:
Feature names for train.csv and test.csv.

fold_N/inner_features.json:
Feature names for inner_train.csv and val.csv.

## Important
Preprocessing is independently fitted using
training data for each split.

The saved CSVs include encounter_id and
patient_nbr for auditing only. These identifiers
must NOT be passed as model input features.

The inner validation files and outer test files
use different training-fitted transformations.
Do not concatenate their encoded features
without appropriate reprocessing.

MEPS is a separate external data source and
is not joined to individual UCI patients.

## Reproduction
Requires the accompanying preprocessing scripts:
run_cdc_xgb.py
run_meps_multitask.py
export_diabetes_5fold.py
"""

    (outdir / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print(
        "\n========== FINISHED ==========",
        flush=True,
    )

    print(
        summary.to_string(index=False),
        flush=True,
    )

    print(
        "\nSaved:", outdir.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()
