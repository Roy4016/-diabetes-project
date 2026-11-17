"""Materialize leakage-safe, patient-disjoint datasets."""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .data import feature_columns, load_uci_diabetes
from .features import add_clinical_features, make_preprocessor, normalize_model_frame
from .splitting import PatientFold, make_patient_folds


def _as_sparse(matrix):
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def _save_partition(directory: Path, name: str, matrix, raw: pd.DataFrame) -> dict:
    sparse.save_npz(directory / f"{name}.npz", _as_sparse(matrix), compressed=True)
    audit_columns = [column for column in ("encounter_id", "patient_nbr", "target") if column in raw]
    raw[audit_columns].to_csv(directory / f"{name}_rows.csv", index=False)
    return {
        "name": name,
        "rows": int(len(raw)),
        "patients": int(raw["patient_nbr"].nunique()),
        "positive_rate": float(raw["target"].mean()),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }


def _write_fold(
    frame: pd.DataFrame,
    columns: list[str],
    fold: PatientFold,
    output_dir: Path,
) -> dict:
    fold_dir = output_dir / f"fold_{fold.number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train, test = frame.iloc[fold.train_idx], frame.iloc[fold.test_idx]
    inner_train, validation = frame.iloc[fold.inner_train_idx], frame.iloc[fold.val_idx]

    train_x = normalize_model_frame(train, columns)
    test_x = normalize_model_frame(test, columns)
    inner_train_x = normalize_model_frame(inner_train, columns)
    validation_x = normalize_model_frame(validation, columns)
    outer = make_preprocessor(train_x, columns)
    x_train = outer.fit_transform(train_x)
    x_test = outer.transform(test_x)
    inner = make_preprocessor(inner_train_x, columns)
    x_inner = inner.fit_transform(inner_train_x)
    x_validation = inner.transform(validation_x)

    joblib.dump({"preprocessor": outer, "feature_columns": columns},
                fold_dir / "outer_preprocessor.joblib", compress=3)
    joblib.dump({"preprocessor": inner, "feature_columns": columns},
                fold_dir / "inner_preprocessor.joblib", compress=3)
    (fold_dir / "outer_feature_names.json").write_text(
        json.dumps(outer.get_feature_names_out().tolist(), indent=2), encoding="utf-8"
    )
    (fold_dir / "inner_feature_names.json").write_text(
        json.dumps(inner.get_feature_names_out().tolist(), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        fold_dir / "indices.npz",
        train_idx=fold.train_idx, test_idx=fold.test_idx,
        inner_train_idx=fold.inner_train_idx, val_idx=fold.val_idx,
    )
    partitions = [
        _save_partition(fold_dir, "train", x_train, train),
        _save_partition(fold_dir, "test", x_test, test),
        _save_partition(fold_dir, "inner_train", x_inner, inner_train),
        _save_partition(fold_dir, "val", x_validation, validation),
    ]
    info = {"fold": fold.number, "patient_overlap": 0, "partitions": partitions}
    (fold_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def preprocess_uci_5fold(
    input_path: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    seed: int = 777,
    validation_size: float = 0.20,
    clinical_features: bool = True,
    overwrite: bool = False,
) -> dict:
    """Create sparse processed matrices and auditable row metadata for every fold."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {destination}. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    frame = load_uci_diabetes(input_path)
    clinical_columns: list[str] = []
    if clinical_features:
        frame, clinical_columns = add_clinical_features(frame)
    columns = feature_columns(frame)
    folds = make_patient_folds(
        frame, n_splits=n_splits, seed=seed, validation_size=validation_size
    )
    fold_info = [_write_fold(frame, columns, fold, destination) for fold in folds]

    assignments = frame[["encounter_id", "patient_nbr", "target"]].copy()
    assignments.insert(0, "row_index", np.arange(len(frame)))
    for fold in folds:
        values = np.full(len(frame), "outer_train", dtype=object)
        values[fold.test_idx] = "test"
        values[fold.inner_train_idx] = "inner_train"
        values[fold.val_idx] = "val"
        assignments[f"fold_{fold.number}"] = values
    assignments.to_csv(destination / "fold_assignments.csv", index=False)

    summary = {
        "dataset": "UCI Diabetes 130-US Hospitals",
        "input": str(Path(input_path).resolve()),
        "rows": int(len(frame)),
        "patients": int(frame["patient_nbr"].nunique()),
        "positive_rate": float(frame["target"].mean()),
        "folds": n_splits,
        "seed": seed,
        "validation_size": validation_size,
        "outer_split": "StratifiedGroupKFold grouped by patient_nbr",
        "inner_split": "GroupShuffleSplit grouped by patient_nbr",
        "source_feature_count": len(columns),
        "clinical_feature_columns": clinical_columns,
        "storage": "SciPy CSR .npz matrices plus *_rows.csv audit metadata",
        "fold_info": fold_info,
    }
    (destination / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary

