"""Harmonized longitudinal MEPS proxies with UCI-targeted domain weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


RACES = ("Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other")
FEATURES = ("age", "male", "prior_er", "prior_inpatient") + tuple(
    f"race_{name}" for name in RACES
)
PROXIES = ("meps_weighted_er", "meps_weighted_ip")


def _age_midpoint(series):
    bounds = series.astype("string").str.extract(r"^\[(\d+)-(\d+)\)$")
    return (
        pd.to_numeric(bounds[0], errors="coerce")
        + pd.to_numeric(bounds[1], errors="coerce")
    ) / 2


def _race_columns(series):
    race = series.astype("string").str.strip()
    race = race.where(race.isin(RACES), "Other").mask(race.isna())
    return {
        f"race_{name}": (race == name).fillna(False).astype(float)
        for name in RACES
    }


def hospital_bridge(frame):
    """Build UCI features that have matching longitudinal MEPS semantics."""
    output = pd.DataFrame(index=frame.index)
    output["age"] = _age_midpoint(frame["age"]).fillna(65).clip(0, 100) / 100
    output["male"] = (
        frame["gender"].astype("string").str.strip().str.lower()
        .map({"male": 1.0, "female": 0.0}).fillna(.5)
    )
    output["prior_er"] = (
        pd.to_numeric(frame["number_emergency"], errors="coerce").fillna(0) > 0
    ).astype(float)
    output["prior_inpatient"] = (
        pd.to_numeric(frame["number_inpatient"], errors="coerce").fillna(0) > 0
    ).astype(float)
    for name, values in _race_columns(frame["race"]).items():
        output[name] = values
    return output.loc[:, FEATURES].astype(np.float32)


def load_longitudinal_pairs(path):
    """Use year-t state and utilization to predict year-(t+1) utilization."""
    frame = pd.read_csv(path, low_memory=False, dtype={"person_key": str})
    required = {
        "person_key", "year", "age", "sex", "race_common", "er_visits",
        "inpatient_nights", "er_any", "inpatient_any", "person_weight",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MEPS data is missing harmonized columns: {sorted(missing)}")
    frame = frame.sort_values(["person_key", "year"]).reset_index(drop=True)
    grouped = frame.groupby("person_key", sort=False)
    next_year = grouped.year.shift(-1)
    valid = next_year.eq(frame.year + 1)

    output = pd.DataFrame(index=frame.index)
    output["person_key"] = frame.person_key
    output["year"] = pd.to_numeric(frame.year, errors="coerce")
    output["age"] = pd.to_numeric(frame.age, errors="coerce").fillna(65).clip(0, 100) / 100
    output["male"] = pd.to_numeric(frame.sex, errors="coerce").map({1: 1.0, 2: 0.0}).fillna(.5)
    output["prior_er"] = (
        pd.to_numeric(frame.er_visits, errors="coerce").fillna(0) > 0
    ).astype(float)
    output["prior_inpatient"] = (
        pd.to_numeric(frame.inpatient_nights, errors="coerce").fillna(0) > 0
    ).astype(float)
    for name, values in _race_columns(frame.race_common).items():
        output[name] = values
    output["target_er"] = pd.to_numeric(grouped.er_any.shift(-1), errors="coerce")
    output["target_ip"] = pd.to_numeric(grouped.inpatient_any.shift(-1), errors="coerce")
    output["survey_weight"] = pd.to_numeric(frame.person_weight, errors="coerce")
    valid &= output[["target_er", "target_ip"]].isin([0, 1]).all(axis=1)
    valid &= output[list(FEATURES)].notna().all(axis=1)
    output = output.loc[valid].reset_index(drop=True)
    if len(output) == 0:
        raise ValueError("No consecutive-year MEPS pairs were available")
    return output


def _external_model(seed):
    return XGBClassifier(
        n_estimators=350, max_depth=3, learning_rate=.04,
        min_child_weight=8, subsample=.85, colsample_bytree=.9,
        reg_lambda=5, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device="cpu", random_state=seed, n_jobs=-1,
    )


def fit_domain_weighted_proxies(pairs, hospital_train, frames, seed, weight_clip=(.1, 10.)):
    """Fit MEPS utilization proxies weighted toward the outer UCI train domain.

    `frames` is a mapping of names to UCI frames that receive proxy columns.
    Only `hospital_train` is used when estimating source-to-target weights.
    """
    source_x = pairs.loc[:, FEATURES].to_numpy(dtype=np.float32)
    target_x = hospital_bridge(hospital_train).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    target_sample = target_x[rng.choice(len(target_x), size=len(source_x), replace=len(target_x) < len(source_x))]
    domain_x = np.vstack([source_x, target_sample])
    domain_y = np.r_[np.zeros(len(source_x)), np.ones(len(target_sample))]
    domain_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=seed),
    ).fit(domain_x, domain_y)
    source_target_probability = np.clip(
        domain_model.predict_proba(source_x)[:, 1], 1e-4, 1 - 1e-4,
    )
    density_ratio = source_target_probability / (1 - source_target_probability)
    density_ratio = np.clip(density_ratio, weight_clip[0], weight_clip[1])

    survey = pd.to_numeric(pairs.survey_weight, errors="coerce").to_numpy(dtype=float)
    survey = np.where(np.isfinite(survey) & (survey > 0), survey, np.nan)
    survey = np.where(np.isfinite(survey), survey / np.nanmean(survey), 1.)
    repeats = pairs.groupby("person_key").person_key.transform("size").to_numpy(dtype=float)
    weights = density_ratio * survey / repeats
    weights /= weights.mean()
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())

    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    train_ids, val_ids = next(split.split(source_x, groups=pairs.person_key))
    diagnostics = {
        "DomainTrainingAUROC": float(roc_auc_score(domain_y, domain_model.predict_proba(domain_x)[:, 1])),
        "DomainWeightMean": float(weights.mean()),
        "DomainWeightP01": float(np.quantile(weights, .01)),
        "DomainWeightP99": float(np.quantile(weights, .99)),
        "DomainWeightMax": float(weights.max()),
        "DomainWeightESS": effective_sample_size,
        "MEPSPairs": int(len(pairs)),
    }
    output = {name: frame.copy() for name, frame in frames.items()}
    for target_number, (target, proxy) in enumerate(zip(("target_er", "target_ip"), PROXIES)):
        y = pairs[target].to_numpy(dtype=int)
        validation_model = _external_model(seed + target_number + 10).fit(
            source_x[train_ids], y[train_ids], sample_weight=weights[train_ids],
        )
        validation_probability = validation_model.predict_proba(source_x[val_ids])[:, 1]
        tag = "ER" if target == "target_er" else "IP"
        diagnostics[f"MEPS_{tag}_ValidationAP"] = float(
            average_precision_score(y[val_ids], validation_probability, sample_weight=weights[val_ids])
        )
        diagnostics[f"MEPS_{tag}_ValidationAUROC"] = float(
            roc_auc_score(y[val_ids], validation_probability, sample_weight=weights[val_ids])
        )
        final_model = _external_model(seed + target_number + 100).fit(
            source_x, y, sample_weight=weights,
        )
        for name, frame in output.items():
            output[name][proxy] = final_model.predict_proba(
                hospital_bridge(frame).to_numpy(dtype=np.float32)
            )[:, 1].astype(np.float32)
    return output, diagnostics
