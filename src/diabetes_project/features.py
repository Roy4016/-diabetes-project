"""Reusable clinical feature engineering and tabular preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DIAGNOSIS_GROUPS = {
    "clinical_hypertension": (401, 405),
    "clinical_lipid_disorder": (272, 272),
    "clinical_ischemic_heart": (410, 414),
    "clinical_heart_failure": (428, 428),
    "clinical_cerebrovascular": (430, 438),
    "clinical_renal_disorder": (580, 589),
    "clinical_copd_related": (490, 496),
    "clinical_liver_disorder": (570, 573),
    "clinical_neoplasm": (140, 208),
}


def diagnosis_flag(frame: pd.DataFrame, low: int, high: int) -> np.ndarray:
    flags = []
    for column in ("diag_1", "diag_2", "diag_3"):
        prefix = frame[column].astype("string").str.strip().str.extract(
            r"^(\d{3})(?:\.|$)", expand=False
        )
        values = pd.to_numeric(prefix, errors="coerce")
        flags.append(values.between(low, high).fillna(False).to_numpy())
    return np.column_stack(flags).any(axis=1)


def add_clinical_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add diagnosis and prior-utilization features used across experiments."""
    output = frame.copy()
    created: list[str] = []
    for name, (low, high) in DIAGNOSIS_GROUPS.items():
        output[name] = diagnosis_flag(output, low, high).astype("int8")
        created.append(name)

    output["clinical_comorbidity_count"] = output[created].sum(axis=1).astype("float32")
    created.append("clinical_comorbidity_count")
    diagnoses = output[["diag_1", "diag_2", "diag_3"]].replace({"": pd.NA})
    output["clinical_unique_diagnoses"] = diagnoses.nunique(axis=1).astype("float32")
    created.append("clinical_unique_diagnoses")

    utilization = []
    for source in ("number_inpatient", "number_emergency", "number_outpatient"):
        if source not in output:
            continue
        values = pd.to_numeric(output[source], errors="coerce").fillna(0).clip(lower=0)
        name = f"clinical_log_{source}"
        output[name] = np.log1p(values).astype("float32")
        created.append(name)
        utilization.append(values)
    if utilization:
        output["clinical_total_prior_visits"] = sum(utilization).astype("float32")
        created.append("clinical_total_prior_visits")
    return output, created


def infer_column_types(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    categorical, numeric = [], []
    categorical_ids = {"admission_type_id", "admission_source_id"}
    for column in columns:
        series = frame[column]
        if (
            column in categorical_ids
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            numeric.append(column)
    return categorical, numeric


def normalize_model_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert pandas extension missing values to sklearn-compatible values."""
    output = frame.loc[:, columns].copy()
    categorical, numeric = infer_column_types(output, columns)
    for column in categorical:
        output[column] = output[column].astype(object)
        output[column] = output[column].where(pd.notna(output[column]), np.nan)
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def make_preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    """Build a preprocessing pipeline whose vocabulary and statistics are fit later."""
    categorical, numeric = infer_column_types(frame, columns)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", encoder),
            ]), categorical),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

