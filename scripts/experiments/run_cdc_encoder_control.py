
import argparse
from pathlib import Path
from _paths import MEPS_BRIDGE, UCI_RAW, result_dir
import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    brier_score_loss,
)

from run_cdc_xgb import load_hospital, make_features


BRIDGE2 = ["bridge_age", "bridge_sex"]

BRIDGE6 = BRIDGE2 + [
    "bridge_high_bp",
    "bridge_high_chol",
    "bridge_stroke",
    "bridge_heart_disease",
]

RACES = [
    "Caucasian",
    "AfricanAmerican",
    "Hispanic",
    "Asian",
    "Other",
]

RACE_COLS = ["bridge_race_" + x for x in RACES]

BRIDGE8 = BRIDGE6 + ["bridge_mi"] + RACE_COLS

PROXIES = ["meps_er_8", "meps_ip_8"]

EXPERIMENTS = {
    "Clinical": [],
    "Direct_Bridge8": BRIDGE8,
    "MEPS_ER_8": ["meps_er_8"],
    "MEPS_IP_8": ["meps_ip_8"],
    "MEPS_Both_8": PROXIES,
    "Direct8_MEPS_Both8": BRIDGE8 + PROXIES,
}

DIAG_GROUPS = {
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

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


# ============================================================
# ICD-9 features
# ============================================================

def icd_flags(df, low, high):
    flags = []

    for c in ["diag_1", "diag_2", "diag_3"]:
        codes = (
            df[c]
            .astype("string")
            .str.strip()
            .str.extract(
                r"^(\d{3})(?:\.|$)",
                expand=False,
            )
        )

        x = pd.to_numeric(codes, errors="coerce")

        flags.append(
            x.between(low, high)
            .fillna(False)
            .to_numpy(dtype=bool)
        )

    return np.column_stack(flags).any(axis=1)


def add_race_onehot(df, race):
    race = race.astype("string").str.strip()

    for c in RACES:
        df["bridge_race_" + c] = np.where(
            race.isna(),
            np.nan,
            (race == c).astype(float),
        ).astype("float32")

    return df


# ============================================================
# Hospital data
# ============================================================

def prepare_hospital(path):
    df = load_hospital(path).reset_index(drop=True)

    for c in BRIDGE2 + [
        "bridge_high_bp", "bridge_high_chol"
    ]:
        if c not in df:
            raise ValueError(
                f"Missing hospital bridge {c}"
            )

    df["bridge_stroke"] = (
        icd_flags(df, 430, 438).astype("int8")
    )

    df["bridge_heart_disease"] = (
        icd_flags(df, 410, 414).astype("int8")
    )

    # Recorded MI diagnosis, not lifetime MI history.
    df["bridge_mi"] = (
        icd_flags(df, 410, 410)
        | icd_flags(df, 412, 412)
    ).astype("int8")

    if "race" not in df:
        raise ValueError("Hospital race field missing")

    r = (
        df["race"]
        .astype("string")
        .str.strip()
        .replace({"?": pd.NA, "": pd.NA})
    )

    r = (
        r.where(r.isin(RACES), "Other")
        .mask(r.isna())
    )

    add_race_onehot(df, r)

    clinical = []

    for name, (lo, hi) in DIAG_GROUPS.items():
        df[name] = (
            icd_flags(df, lo, hi).astype("int8")
        )
        clinical.append(name)

    complication = np.zeros(
        len(df), dtype=bool
    )

    for col in ["diag_1", "diag_2", "diag_3"]:
        complication |= (
            df[col]
            .astype("string")
            .str.strip()
            .str.match(
                r"^250\.[1-9][0-9]?$",
                na=False,
            )
            .to_numpy(dtype=bool)
        )

    df["clinical_diabetes_complication"] = (
        complication.astype("int8")
    )

    clinical.append(
        "clinical_diabetes_complication"
    )

    df["clinical_comorbidity_count"] = (
        df[clinical]
        .sum(axis=1)
        .astype("float32")
    )

    clinical.append("clinical_comorbidity_count")

    diagnosis = (
        df[["diag_1", "diag_2", "diag_3"]]
        .astype("string")
        .replace({"?": pd.NA, "": pd.NA})
    )

    df["clinical_unique_diagnoses"] = (
        diagnosis.nunique(axis=1)
        .astype("float32")
    )

    clinical.append("clinical_unique_diagnoses")

    meds = (
        df[[c for c in MEDICATIONS if c in df]]
        .fillna("No")
        .astype(str)
    )

    if meds.empty:
        raise ValueError(
            "No medication fields found"
        )

    med_features = {
        "clinical_active_diabetes_drugs":
            meds.isin(
                ["Steady", "Up", "Down"]
            ).sum(axis=1),

        "clinical_drugs_increased":
            meds.eq("Up").sum(axis=1),

        "clinical_drugs_decreased":
            meds.eq("Down").sum(axis=1),

        "clinical_drugs_adjusted":
            meds.isin(
                ["Up", "Down"]
            ).sum(axis=1),
    }

    for name, values in med_features.items():
        df[name] = values.astype("float32")
        clinical.append(name)

    util = pd.DataFrame(index=df.index)

    for c in [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]:
        util[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )

        name = "clinical_log_" + c

        df[name] = np.log1p(
            util[c]
        ).astype("float32")

        clinical.append(name)

    df["clinical_total_prior_visits"] = (
        util.sum(axis=1).astype("float32")
    )

    df["clinical_prior_inpatient_flag"] = (
        util["number_inpatient"] > 0
    ).astype("int8")

    df["clinical_repeated_inpatient_flag"] = (
        util["number_inpatient"] >= 2
    ).astype("int8")

    df["clinical_prior_emergency_flag"] = (
        util["number_emergency"] > 0
    ).astype("int8")

    clinical.extend([
        "clinical_total_prior_visits",
        "clinical_prior_inpatient_flag",
        "clinical_repeated_inpatient_flag",
        "clinical_prior_emergency_flag",
    ])

    df["clinical_comorbidity_x_prior_visits"] = (
        df["clinical_comorbidity_count"]
        * np.log1p(
            df["clinical_total_prior_visits"]
        )
    ).astype("float32")

    clinical.append(
        "clinical_comorbidity_x_prior_visits"
    )

    print(
        f"[Hospital] Rows={len(df):,}; "
        f"Patients={df.patient_nbr.nunique():,}; "
        f"Clinical={len(clinical)}",
        flush=True,
    )

    return df, clinical


# ============================================================
# MEPS data
# ============================================================

def load_meps(path):
    df = pd.read_csv(path, low_memory=False)

    needed = [
        "person_key", "age", "sex",
        "high_bp", "high_chol",
        "stroke", "heart_disease",
        "race_common", "mi",
        "er_any", "inpatient_any",
    ]

    missing = [x for x in needed if x not in df]

    if missing:
        raise ValueError(
            f"Missing MEPS columns: {missing}"
        )

    df["person_key"] = df["person_key"].astype(str)

    for c in needed:
        if c not in ["person_key", "race_common"]:
            df[c] = pd.to_numeric(
                df[c], errors="coerce"
            )

    # MEPS SEX: 1=male, 2=female.
    if not df["sex"].dropna().isin([1, 2]).all():
        raise ValueError(
            "Unexpected MEPS SEX codes"
        )

    df["bridge_age"] = (
        df["age"]
        .mask(df["age"] < 0)
        .astype("float32")
    )

    df["bridge_sex"] = (
        df["sex"]
        .map({1: 1.0, 2: 0.0})
        .astype("float32")
    )

    mapping = [
        ("bridge_high_bp", "high_bp"),
        ("bridge_high_chol", "high_chol"),
        ("bridge_stroke", "stroke"),
        ("bridge_heart_disease", "heart_disease"),
        ("bridge_mi", "mi"),
    ]

    for dst, src in mapping:
        df[dst] = df[src].astype("float32")

    add_race_onehot(df, df["race_common"])

    for col in ["er_any", "inpatient_any"]:
        if (
            not df[col].dropna().isin([0, 1]).all()
            or df[col].nunique() != 2
        ):
            raise ValueError(
                f"MEPS {col} not valid binary target"
            )

    print(
        f"[MEPS] Rows={len(df):,}; "
        f"Persons={df.person_key.nunique():,}; "
        f"ER={df.er_any.mean():.4f}; "
        f"IP={df.inpatient_any.mean():.4f}",
        flush=True,
    )

    return df


# ============================================================
# XGBoost models
# ============================================================

def model(seed, external=False, device="cpu", params=None,
          estimators=None, early_stopping=None):
    if external:
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=-1,
        )

    params = params or {}
    return XGBClassifier(
        n_estimators=estimators or 400,
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.04),
        min_child_weight=params.get("min_child_weight", 5),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=params.get("reg_lambda", 3),
        reg_alpha=params.get("reg_alpha", 0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        objective="binary:logistic",
        eval_metric="aucpr" if early_stopping else "logloss",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
    )


TUNING_CANDIDATES = [
    {"max_depth": 3, "learning_rate": 0.04, "min_child_weight": 5,
     "scale_pos_weight": 1.0, "reg_lambda": 3},
    {"max_depth": 4, "learning_rate": 0.04, "min_child_weight": 5,
     "scale_pos_weight": 1.0, "reg_lambda": 3},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5,
     "scale_pos_weight": 2.0, "reg_lambda": 5},
    {"max_depth": 5, "learning_rate": 0.025, "min_child_weight": 8,
     "scale_pos_weight": 2.0, "reg_lambda": 5},
]


def tune_model(x_train, y_train, x_validation, y_validation, seed, device):
    """Select XGBoost settings using only the inner validation partition."""
    candidates = []
    for index, params in enumerate(TUNING_CANDIDATES):
        fitted = model(
            seed + index * 101, device=device, params=params,
            estimators=1000, early_stopping=40,
        )
        fitted.fit(
            x_train, y_train,
            eval_set=[(x_validation, y_validation)], verbose=False,
        )
        probability = fitted.predict_proba(x_validation)[:, 1]
        trees = int(getattr(fitted, "best_iteration", 999)) + 1
        candidates.append({
            "params": params, "trees": trees,
            "validation_ap": average_precision_score(y_validation, probability),
            "probability": probability,
        })
        print(
            f"[Tune] candidate={index + 1}/{len(TUNING_CANDIDATES)} "
            f"validation_AP={candidates[-1]['validation_ap']:.5f} "
            f"trees={trees} params={params}",
            flush=True,
        )
    return max(candidates, key=lambda item: item["validation_ap"])


# ============================================================
# Train MEPS external models
# ============================================================

def external_meps(meps, hospital, seed, outdir, device="cpu"):
    hospital = hospital.copy()

    print(
        "\n========== BRIDGE DIAGNOSTICS ==========",
        flush=True,
    )

    for c in BRIDGE6 + ["bridge_mi"]:
        print(
            f"{c:25} "
            f"MEPS={meps[c].mean():.4f} "
            f"Hospital={hospital[c].mean():.4f}",
            flush=True,
        )

    for c in RACE_COLS:
        print(
            f"{c:25} "
            f"MEPS={meps[c].mean():.4f} "
            f"Hospital={hospital[c].mean():.4f}",
            flush=True,
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )

    tr, va = next(
        splitter.split(
            meps,
            groups=meps.person_key,
        )
    )

    rows = []

    print(
        "\n========== EXTERNAL MEPS VALIDATION ==========",
        flush=True,
    )

    configurations = [
        (2, BRIDGE2),
        (6, BRIDGE6),
        (8, BRIDGE8),
    ]

    for n, cols in configurations:
        for tag, target in [
            ("ER", "er_any"),
            ("IP", "inpatient_any"),
        ]:
            valid = (
                meps[cols].notna().all(axis=1)
                & meps[target].notna()
            )

            a = meps.iloc[tr].loc[
                valid.iloc[tr]
            ]

            b = meps.iloc[va].loc[
                valid.iloc[va]
            ]

            ytr = a[target].astype(int)
            yva = b[target].astype(int)

            if (
                ytr.nunique() < 2
                or yva.nunique() < 2
            ):
                raise ValueError(
                    f"One-class MEPS {tag}-{n}"
                )

            counts = a.groupby(
                "person_key"
            )["person_key"].transform("size")

            seed_this = (
                seed + n + (0 if tag == "ER" else 100)
            )

            m = model(seed_this, external=True, device=device)

            m.fit(
                a[cols].astype("float32"),
                ytr,
                sample_weight=(1 / counts).to_numpy(),
            )

            p = m.predict_proba(
                b[cols].astype("float32")
            )[:, 1]

            ap = average_precision_score(yva, p)
            auc = roc_auc_score(yva, p)

            rows.append({
                "Model": f"MEPS_{tag}_{n}",
                "ValidationRows": len(b),
                "ValidationPatients":
                    b.person_key.nunique(),
                "Prevalence": yva.mean(),
                "AP": ap,
                "AUROC": auc,
            })

            print(
                f"[MEPS] {tag}-{n} "
                f"AP={ap:.5f} "
                f"AUROC={auc:.5f} "
                f"rows={len(b)}",
                flush=True,
            )

            # Fit transfer models with all MEPS rows.
            if n == 8:
                full = meps.loc[valid]

                full_counts = full.groupby(
                    "person_key"
                )["person_key"].transform("size")

                full_model = model(
                    seed_this, external=True, device=device
                )

                full_model.fit(
                    full[cols].astype("float32"),
                    full[target].astype(int),
                    sample_weight=(
                        1 / full_counts
                    ).to_numpy(),
                )

                cname = f"meps_{tag.lower()}_8"

                hospital[cname] = (
                    full_model.predict_proba(
                        hospital[cols].astype("float32")
                    )[:, 1].astype("float32")
                )

    pd.DataFrame(rows).to_csv(
        outdir / "meps_external_validation.csv",
        index=False,
    )

    return hospital


# ============================================================
# Hospital feature preparation
# ============================================================

def feature_columns(df, clinical, experiment):
    excluded = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target",
        *BRIDGE8,
        *PROXIES,
        *clinical,
    }

    original = [
        c for c in df.columns
        if c not in excluded
    ]

    return (
        original
        + clinical
        + EXPERIMENTS[experiment]
    )


def encode(tr, va, te, cols):
    matrices = make_features(
        tr, va, te, cols
    )

    return [
        m.replace(
            [np.inf, -np.inf], np.nan
        ).astype("float32")
        for m in matrices
    ]


def threshold(y, p):
    ts = np.arange(0.05, 0.6001, 0.005)

    scores = [
        f1_score(
            y, p >= t, zero_division=0
        )
        for t in ts
    ]

    return float(
        ts[int(np.argmax(scores))]
    )


# ============================================================
# Patient-disjoint Hospital CV
# ============================================================

def hospital_cv(df, clinical, folds, seed, outdir, device="cpu",
                nested_tune=False):
    rows = []
    oofs = []

    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    splits = cv.split(
        df,
        df.target,
        groups=df.patient_nbr,
    )

    for fold, (itr, ite) in enumerate(splits, 1):
        train = df.iloc[itr].copy()
        test = df.iloc[ite].copy()

        assert set(train.patient_nbr).isdisjoint(
            set(test.patient_nbr)
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=seed + fold,
        )

        a, b = next(
            splitter.split(
                train,
                groups=train.patient_nbr,
            )
        )

        inner = train.iloc[a]
        val = train.iloc[b]

        if any(
            part.target.nunique() != 2
            for part in (inner, val, train, test)
        ):
            raise ValueError(
                "One-class hospital split"
            )

        print(
            f"\n========== HOSPITAL FOLD "
            f"{fold}/{folds} ==========",
            flush=True,
        )

        oof = (
            test[
                ["encounter_id", "patient_nbr", "target"]
            ]
            .copy()
            .rename(columns={"target": "y_true"})
        )

        oof["Fold"] = fold

        for exp in EXPERIMENTS:
            cols = feature_columns(
                df, clinical, exp
            )

            # Inner validation: select F1 threshold.
            xi, xv, _ = encode(
                inner, val, val, cols
            )

            selected = None
            # Keep the established Clinical baseline fixed. Tune only the
            # proposed external-data model, which is the model under study.
            if nested_tune and exp != "Clinical":
                selected = tune_model(
                    xi, inner.target.to_numpy(), xv,
                    val.target.to_numpy(), seed + fold, device,
                )
                valp = selected["probability"]
            else:
                m = model(seed + fold, device=device)
                m.fit(xi, inner.target)
                valp = m.predict_proba(xv)[:, 1]

            t = threshold(
                val.target.to_numpy(),
                valp,
            )

            # Refit on full outer training fold.
            xt, _, xh = encode(
                train, train, test, cols
            )

            m = model(
                seed + fold, device=device,
                params=selected["params"] if selected else None,
                estimators=selected["trees"] if selected else None,
            )
            m.fit(xt, train.target)

            p = m.predict_proba(xh)[:, 1]
            y = test.target.to_numpy()

            ap = average_precision_score(y, p)
            auc = roc_auc_score(y, p)

            f1 = f1_score(
                y, p >= t, zero_division=0
            )

            brier = brier_score_loss(y, p)

            rows.append({
                "Fold": fold,
                "Experiment": exp,
                "Features": xt.shape[1],
                "AP": ap,
                "AUROC": auc,
                "F1": f1,
                "Brier": brier,
                "Threshold": t,
                "ValidationAP": (
                    selected["validation_ap"] if selected else
                    average_precision_score(val.target, valp)
                ),
                "Trees": selected["trees"] if selected else 400,
                "TuningParams": json.dumps(
                    selected["params"] if selected else {}, sort_keys=True
                ),
            })

            oof["p_" + exp] = p

            print(
                f"[Done] Fold={fold} "
                f"{exp:24} "
                f"AP={ap:.5f} "
                f"AUROC={auc:.5f} "
                f"F1={f1:.5f}",
                flush=True,
            )

        oofs.append(oof)

        pd.DataFrame(rows).to_csv(
            outdir / "partial_results.csv",
            index=False,
        )

    return (
        pd.DataFrame(rows),
        pd.concat(oofs, ignore_index=True),
    )


# ============================================================
# Patient-level paired bootstrap
# ============================================================

def paired_bootstrap(oof, n, seed, outdir):
    patients, inverse = np.unique(
        oof.patient_nbr.to_numpy(),
        return_inverse=True,
    )

    y = oof.y_true.to_numpy()
    rng = np.random.default_rng(seed + 9000)

    reference = oof.p_Clinical.to_numpy()
    comparisons = []

    for exp in EXPERIMENTS:
        if exp == "Clinical":
            continue

        p = oof["p_" + exp].to_numpy()

        observed = (
            average_precision_score(y, p)
            - average_precision_score(y, reference)
        )

        draws = []

        for _ in range(n):
            counts = np.bincount(
                rng.integers(
                    0,
                    len(patients),
                    size=len(patients),
                ),
                minlength=len(patients),
            )

            w = counts[inverse]

            draws.append(
                average_precision_score(
                    y, p, sample_weight=w
                )
                - average_precision_score(
                    y, reference, sample_weight=w
                )
            )

        draws = np.asarray(draws)

        comparisons.append({
            "Comparison": f"{exp} - Clinical",
            "PooledOOF_DeltaAP": observed,
            "CI95_Lower": np.quantile(
                draws, 0.025
            ),
            "CI95_Upper": np.quantile(
                draws, 0.975
            ),
            "FractionPositive":
                (draws > 0).mean(),
        })

    table = pd.DataFrame(comparisons)

    table.to_csv(
        outdir / "patient_bootstrap.csv",
        index=False,
    )

    print(
        "\n========== PATIENT BOOTSTRAP ==========",
        flush=True,
    )

    print(
        table.round(6).to_string(index=False),
        flush=True,
    )


# ============================================================
# Final summary
# ============================================================

def summarize(results, oof, bootstrap, seed, outdir):
    results.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    oof.to_csv(
        outdir / "oof_all.csv",
        index=False,
    )

    stats = results.groupby(
        "Experiment", sort=False
    )[["AP", "AUROC", "F1", "Brier"]].agg(
        ["mean", "std"]
    )

    stats.columns = [
        f"{a}_{b}"
        for a, b in stats.columns
    ]

    stats = stats.reset_index()

    stats.to_csv(
        outdir / "summary.csv",
        index=False,
    )

    print(
        "\n========== FINAL 5-FOLD SUMMARY ==========",
        flush=True,
    )

    print(
        stats.round(6).to_string(index=False),
        flush=True,
    )

    pivot = results.pivot(
        index="Fold",
        columns="Experiment",
        values="AP",
    )

    delta = []

    for exp in EXPERIMENTS:
        if exp == "Clinical":
            continue

        d = pivot[exp] - pivot["Clinical"]

        pooled_delta = (
            average_precision_score(
                oof.y_true,
                oof["p_" + exp],
            )
            - average_precision_score(
                oof.y_true,
                oof.p_Clinical,
            )
        )

        delta.append({
            "Comparison": f"{exp} - Clinical",
            "MeanDeltaAP": d.mean(),
            "StdDeltaAP": d.std(),
            "PositiveFolds": int((d > 0).sum()),
            "PooledOOF_DeltaAP": pooled_delta,
        })

    table = pd.DataFrame(delta)

    table.to_csv(
        outdir / "paired_deltas.csv",
        index=False,
    )

    print(
        "\n========== MEPS VS CLINICAL ==========",
        flush=True,
    )

    print(
        table.round(6).to_string(index=False),
        flush=True,
    )

    if bootstrap > 0:
        paired_bootstrap(
            oof, bootstrap, seed, outdir
        )


# ============================================================
# Main
# ============================================================

def main():
    global EXPERIMENTS
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital",
        default=UCI_RAW,
    )

    parser.add_argument(
        "--meps",
        default=MEPS_BRIDGE,
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

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )

    parser.add_argument(
        "--nested-tune",
        action="store_true",
        help="Select XGBoost settings on each fold's inner validation split.",
    )

    parser.add_argument(
        "--outdir",
        default=result_dir("meps_bridge8_results"),
    )

    args = parser.parse_args()

    if args.folds < 2 or args.bootstrap < 0:
        parser.error(
            "folds >= 2 and bootstrap >= 0 required"
        )
    if "Clinical" not in args.experiments:
        parser.error("--experiments must include Clinical")
    EXPERIMENTS = {
        name: EXPERIMENTS[name]
        for name in args.experiments
    }

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (outdir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    hospital, clinical = prepare_hospital(
        args.hospital
    )

    meps = load_meps(args.meps)

    hospital = external_meps(
        meps,
        hospital,
        args.seed,
        outdir,
        args.device,
    )

    results, oof = hospital_cv(
        hospital,
        clinical,
        args.folds,
        args.seed,
        outdir,
        args.device,
        args.nested_tune,
    )

    summarize(
        results,
        oof,
        args.bootstrap,
        args.seed,
        outdir,
    )

    print(
        "\nFINISHED:",
        outdir.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()
