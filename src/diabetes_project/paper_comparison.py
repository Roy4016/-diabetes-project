"""Shared cohort, metrics, and SHAP helpers for the paper comparison.

The 46-field recoding follows the MIT-licensed Bhuvan Rajpoot reproduction
(`Bhuvan_Rajpoot_SSRN/src/readmission/data.py`, inspected 2026-11-21).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
NUMERIC = {"age", "time_in_hospital", "num_lab_procedures", "num_procedures",
           "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "number_diagnoses"}
CHAPTERS = (
    (1, 139, "infectious_parasitic"), (140, 239, "neoplasms"),
    (240, 279, "endocrine_metabolic"), (280, 289, "blood"),
    (290, 319, "mental"), (320, 389, "nervous_sense"),
    (390, 459, "circulatory"), (460, 519, "respiratory"),
    (520, 579, "digestive"), (580, 629, "genitourinary"),
    (630, 679, "pregnancy_childbirth"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"),
    (760, 779, "perinatal"), (780, 799, "symptoms_ill_defined"),
    (800, 999, "injury_poisoning"),
)


def load_comparison_cohort(path):
    """Use the 99,343-encounter cohort in the latest five-fold HGB work."""
    source = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)
    source.columns = source.columns.str.strip()
    required = {"encounter_id", "patient_nbr", "readmitted", "discharge_disposition_id"}
    if missing := required - set(source):
        raise ValueError(f"Missing UCI columns: {sorted(missing)}")
    valid = source.readmitted.isin(["<30", ">30", "NO"])
    discharge = pd.to_numeric(source.discharge_disposition_id, errors="coerce")
    keep = valid & source.patient_nbr.notna() & ~discharge.isin(EXCLUDED_DISPOSITIONS)
    cohort = source.loc[keep].copy().reset_index(drop=True)
    cohort["target"] = cohort.readmitted.eq("<30").astype(int)
    if cohort.encounter_id.duplicated().any() or cohort.patient_nbr.isna().any():
        raise ValueError("Duplicate encounters or missing patients")
    return cohort


def _diagnosis_family(value):
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return np.nan
    value = str(value).strip().upper()
    if value.startswith("V"):
        return "supplemental_V"
    if value.startswith("E"):
        return "external_E"
    try:
        code = int(float(value))
    except ValueError:
        return "other"
    return next((name for lo, hi, name in CHAPTERS if lo <= code <= hi), "other")


def make_paper_catboost_features(cohort, raw_icd=False):
    """Return the Rajpoot reproduction's 46 discharge predictors for CatBoost."""
    excluded = {"encounter_id", "patient_nbr", "readmitted", "weight", "target"}
    columns = [c for c in cohort if c not in excluded]
    if len(columns) != 46:
        raise ValueError(f"Expected 46 predictors, found {len(columns)}")
    x = cohort.loc[:, columns].copy()
    x["age"] = x.age.map(lambda v: sum(map(int, re.fullmatch(r"\[(\d+)-(\d+)\)", str(v)).groups())) / 2)
    if not raw_icd:
        for column in ("diag_1", "diag_2", "diag_3"):
            x[column] = x[column].map(_diagnosis_family)
    categorical = [column for column in columns if column not in NUMERIC]
    for column in categorical:
        x[column] = x[column].astype(object).where(x[column].notna(), "__MISSING__").astype(str)
    for column in NUMERIC:
        x[column] = pd.to_numeric(x[column], errors="raise")
    return x, categorical


def top_fraction_metrics(y, probability, fraction=0.10):
    """Prespecified alert budget; ranking uses scores only, never test labels."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0 or not np.isfinite(probability).all() or len(np.unique(y)) < 2:
        raise ValueError("Metrics require finite scores and both classes")
    k = max(1, int(np.floor(len(y) * fraction)))
    alerted = np.zeros(len(y), dtype=bool)
    alerted[np.argsort(-probability, kind="mergesort")[:k]] = True
    tp = int(np.sum(alerted & (y == 1)))
    precision = tp / k
    recall = tp / int(y.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "PR_AUC": float(average_precision_score(y, probability)),
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1,
        "Brier_Score": float(brier_score_loss(y, probability)),
        "Alerted": k,
        "True_Positive": tp,
        "Positive_Rate": float(y.mean()),
    }


def catboost_shap_sample(model, x, categorical, sample_size, seed):
    """Exact CatBoost TreeSHAP on raw log-odds, as in the Salim analysis."""
    from catboost import Pool

    if sample_size > 0 and len(x) > sample_size:
        chosen = np.sort(np.random.default_rng(seed).choice(len(x), sample_size, replace=False))
        x = x.iloc[chosen]
    pool = Pool(x, cat_features=categorical)
    values = model.get_feature_importance(pool, type="ShapValues")
    raw_score = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"))
    if not np.allclose(values.sum(axis=1), raw_score, rtol=1e-6, atol=1e-6):
        raise ValueError("CatBoost SHAP values do not sum to model raw scores")
    return pd.DataFrame({
        "Feature": x.columns,
        "MeanAbsSHAP_logodds": np.abs(values[:, :-1]).mean(axis=0),
        "MeanSHAP_logodds": values[:, :-1].mean(axis=0),
        "Explained_Rows": len(x),
    })


def hgb_shap_sample(model, encoded, transformer, raw_columns, sample_size, seed):
    """TreeSHAP for HGB, summing one-hot contributions to original fields first."""
    import shap

    if sample_size > 0 and len(encoded) > sample_size:
        chosen = np.sort(np.random.default_rng(seed).choice(len(encoded), sample_size, replace=False))
        encoded = encoded[chosen]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(encoded))
    if values.ndim != 2 or values.shape != encoded.shape:
        raise ValueError(f"Unexpected HGB SHAP shape: {values.shape}")
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw_score = model.decision_function(encoded)
    if not np.allclose(base + values.sum(axis=1), raw_score, rtol=1e-5, atol=1e-5):
        raise ValueError("HGB SHAP values do not sum to model raw scores")
    names = transformer.get_feature_names_out()
    groups = []
    sorted_columns = sorted(raw_columns, key=len, reverse=True)
    for name in names:
        if name.startswith("numeric__"):
            feature = name.removeprefix("numeric__")
        elif name.startswith("categorical__"):
            suffix = name.removeprefix("categorical__")
            feature = next((col for col in sorted_columns if suffix == col or suffix.startswith(col + "_")), None)
        else:
            feature = None
        if feature not in raw_columns:
            raise ValueError(f"Cannot map encoded SHAP feature {name} to an input field")
        groups.append(feature)
    collapsed = np.column_stack([values[:, np.array(groups) == col].sum(axis=1) for col in raw_columns])
    return pd.DataFrame({
        "Feature": raw_columns,
        "MeanAbsSHAP_logodds": np.abs(collapsed).mean(axis=0),
        "MeanSHAP_logodds": collapsed.mean(axis=0),
        "Explained_Rows": len(encoded),
    })
