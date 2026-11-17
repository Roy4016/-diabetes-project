
import argparse
from pathlib import Path
from _paths import MEPS_BRIDGE, UCI_RAW, result_dir

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


RACES = [
    "Caucasian",
    "AfricanAmerican",
    "Hispanic",
    "Asian",
    "Other",
]

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

STAT_GROUPS = {
    "ERIP": [
        "meps_er_rate",
        "meps_ip_rate",
    ],
    "Health": [
        "meps_physical_health",
        "meps_mental_health",
        "meps_general_health",
        "meps_activity_limitation",
    ],
    "Util": [
        "meps_er_visits",
        "meps_inpatient_nights",
        "meps_log_total_expense",
    ],
}

ALL_STATS = [
    c for values in STAT_GROUPS.values()
    for c in values
]

DIRECT_COLS = [
    "direct_age_group",
    "direct_sex",
] + [
    "direct_race_" + race
    for race in RACES
]

EXPERIMENTS = {
    "Clinical": [],
    "Direct_Demog": DIRECT_COLS,
    "Cell_ERIP": STAT_GROUPS["ERIP"],
    "Cell_Health": STAT_GROUPS["Health"],
    "Cell_Util": STAT_GROUPS["Util"],
    "Cell_All": ALL_STATS,
    "Direct_Demog_Cell_All":
        DIRECT_COLS + ALL_STATS,
}


# ============================================================
# Clinical features: same 24 features as previous experiment
# ============================================================

def icd_flags(df, low, high):
    result = []

    for c in ["diag_1", "diag_2", "diag_3"]:
        code = (
            df[c].astype("string")
            .str.strip()
            .str.extract(
                r"^(\d{3})(?:\.|$)",
                expand=False,
            )
        )

        code = pd.to_numeric(
            code, errors="coerce"
        )

        result.append(
            code.between(low, high)
            .fillna(False)
            .to_numpy(dtype=bool)
        )

    return np.column_stack(result).any(axis=1)


def add_clinical(df):
    df = df.copy()
    clinical = []

    for name, (lo, hi) in DIAG_GROUPS.items():
        df[name] = (
            icd_flags(df, lo, hi)
            .astype("int8")
        )
        clinical.append(name)

    comp = np.zeros(len(df), dtype=bool)

    for c in ["diag_1", "diag_2", "diag_3"]:
        comp |= (
            df[c].astype("string")
            .str.strip()
            .str.match(
                r"^250\.[1-9][0-9]?$",
                na=False,
            )
            .to_numpy(dtype=bool)
        )

    df["clinical_diabetes_complication"] = (
        comp.astype("int8")
    )
    clinical.append(
        "clinical_diabetes_complication"
    )

    df["clinical_comorbidity_count"] = (
        df[clinical].sum(axis=1)
        .astype("float32")
    )
    clinical.append(
        "clinical_comorbidity_count"
    )

    diag = (
        df[["diag_1", "diag_2", "diag_3"]]
        .astype("string")
        .replace({"?": pd.NA, "": pd.NA})
    )

    df["clinical_unique_diagnoses"] = (
        diag.nunique(axis=1)
        .astype("float32")
    )
    clinical.append(
        "clinical_unique_diagnoses"
    )

    medcols = [
        c for c in MEDICATIONS if c in df
    ]

    if not medcols:
        raise ValueError(
            "Hospital medication columns missing"
        )

    meds = (
        df[medcols].fillna("No").astype(str)
    )

    features = {
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

    for name, values in features.items():
        df[name] = values.astype("float32")
        clinical.append(name)

    util = pd.DataFrame(index=df.index)

    for c in [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]:
        util[c] = (
            pd.to_numeric(
                df[c], errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
        )

        name = "clinical_log_" + c
        df[name] = np.log1p(
            util[c]
        ).astype("float32")
        clinical.append(name)

    df["clinical_total_prior_visits"] = (
        util.sum(axis=1)
        .astype("float32")
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

    clinical += [
        "clinical_total_prior_visits",
        "clinical_prior_inpatient_flag",
        "clinical_repeated_inpatient_flag",
        "clinical_prior_emergency_flag",
    ]

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
        f"[Clinical] Features={len(clinical)}",
        flush=True,
    )

    return df, clinical


# ============================================================
# Common demographic cells
# ============================================================

def make_age_group(x):
    x = pd.to_numeric(
        x, errors="coerce"
    )

    x = x.mask(x < 0)

    # 0-9 -> 0, 10-19 -> 1, ...,
    # 90+ -> 9.
    return (
        np.floor(x.clip(0, 99) / 10)
        .fillna(-1)
        .astype(int)
    )


def add_cells(df, age_col, sex_col, race_col):
    df = df.copy()

    df["cell_age"] = make_age_group(
        df[age_col]
    )

    sex = pd.to_numeric(
        df[sex_col], errors="coerce"
    )

    df["cell_sex"] = (
        sex.fillna(-1).astype(int)
    )

    race = (
        df[race_col]
        .astype("string")
        .str.strip()
    )

    race = race.where(
        race.isin(RACES),
        "Unknown",
    )

    df["cell_race"] = race.fillna(
        "Unknown"
    ).astype(str)

    return df


def add_direct_features(df):
    df["direct_age_group"] = (
        df["cell_age"]
        .replace(-1, np.nan)
        .astype("float32")
    )

    df["direct_sex"] = (
        df["cell_sex"]
        .replace(-1, np.nan)
        .astype("float32")
    )

    for race in RACES:
        df["direct_race_" + race] = (
            df["cell_race"] == race
        ).astype("float32")

    return df


# ============================================================
# MEPS preparation
# ============================================================

def load_meps(path):
    df = pd.read_csv(
        path, low_memory=False
    )

    needed = [
        "person_key", "age", "sex",
        "race_common",
        "er_any", "inpatient_any",
        "er_visits", "inpatient_nights",
        "physical_health", "mental_health",
        "general_health",
        "activity_limitation",
        "total_expense",
    ]

    missing = [
        c for c in needed if c not in df
    ]

    if missing:
        raise ValueError(
            f"MEPS columns missing: {missing}"
        )

    df["person_key"] = (
        df["person_key"].astype(str)
    )

    numeric = [
        c for c in needed
        if c not in [
            "person_key", "race_common"
        ]
    ]

    for c in numeric:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

        # Handle SAS XPORT pseudo-zero artifacts.
        df.loc[
            df[c].abs() < 1e-30, c
        ] = 0

    # MEPS sex: 1=Male, 2=Female
    if not df["sex"].dropna().isin(
        [1, 2]
    ).all():
        raise ValueError(
            "Unexpected MEPS sex coding"
        )

    df["sex_common"] = (
        df["sex"]
        .map({1: 1, 2: 0})
    )

    df = add_cells(
        df,
        "age",
        "sex_common",
        "race_common",
    )

    for c in [
        "er_any", "inpatient_any"
    ]:
        if not df[c].dropna().isin(
            [0, 1]
        ).all():
            raise ValueError(
                f"Invalid target: {c}"
            )

    nonnegative = [
        "er_visits",
        "inpatient_nights",
        "total_expense",
    ]

    for c in nonnegative:
        df.loc[df[c] < 0, c] = np.nan

    df["log_total_expense"] = np.log1p(
        df["total_expense"]
    )

    # MEPS general health typically 1..5;
    # activity limitation typically 1/2.
    df.loc[
        ~df["general_health"].between(1, 5),
        "general_health",
    ] = np.nan

    df.loc[
        ~df["activity_limitation"].isin([1, 2]),
        "activity_limitation",
    ] = np.nan

    # 1 = activity limitation, 0 = no limitation
    df["activity_limit_flag"] = (
        df["activity_limitation"]
        .map({1: 1.0, 2: 0.0})
    )

    print(
        f"[MEPS] Rows={len(df):,}; "
        f"Persons={df.person_key.nunique():,}",
        flush=True,
    )

    print(
        "[MEPS] ER rate="
        f"{df.er_any.mean():.4f}; "
        "IP rate="
        f"{df.inpatient_any.mean():.4f}",
        flush=True,
    )

    return df


# ============================================================
# Hospital preparation
# ============================================================

def load_uci(path):
    df = load_hospital(path)
    df = df.reset_index(drop=True)

    df, clinical = add_clinical(df)

    df = add_cells(
        df,
        "bridge_age",
        "bridge_sex",
        "race",
    )

    df = add_direct_features(df)

    print(
        f"[Hospital] Rows={len(df):,}; "
        f"Patients={df.patient_nbr.nunique():,}; "
        f"Readmission={df.target.mean():.4f}",
        flush=True,
    )

    return df, clinical


# ============================================================
# Build external cell statistics
# ============================================================

METRICS = {
    "meps_er_rate": "er_any",
    "meps_ip_rate": "inpatient_any",
    "meps_physical_health": "physical_health",
    "meps_mental_health": "mental_health",
    "meps_general_health": "general_health",
    "meps_activity_limitation":
        "activity_limit_flag",
    "meps_er_visits": "er_visits",
    "meps_inpatient_nights":
        "inpatient_nights",
    "meps_log_total_expense":
        "log_total_expense",
}

LEVELS = [
    ["cell_age"],
    ["cell_age", "cell_sex"],
    ["cell_age", "cell_sex", "cell_race"],
]


def group_statistics(meps, keys, source, weights):
    data = meps[keys].copy()

    values = pd.to_numeric(
        meps[source], errors="coerce"
    )

    valid = values.notna()

    data["value_weight"] = np.where(
        valid,
        values.fillna(0).to_numpy() * weights,
        0.0,
    )

    data["valid_weight"] = np.where(
        valid, weights, 0.0
    )

    data["valid_n"] = valid.astype(int)

    grouped = (
        data.groupby(keys, dropna=False)
        .agg(
            weighted_sum=(
                "value_weight", "sum"
            ),
            weighted_n=(
                "valid_weight", "sum"
            ),
            n=("valid_n", "sum"),
        )
        .reset_index()
    )

    grouped["mean"] = (
        grouped["weighted_sum"]
        / grouped["weighted_n"].replace(
            0, np.nan
        )
    )

    return grouped


def attach_cell_stat(
    meps,
    hospital,
    source,
    min_cell,
    alpha,
):
    # Count a participant with multiple person-years
    # less heavily than multiple independent people.
    person_count = (
        meps.groupby("person_key")[
            "person_key"
        ].transform("size")
    )

    weights = (
        1.0 / person_count.to_numpy()
    )

    values = pd.to_numeric(
        meps[source], errors="coerce"
    )

    valid = values.notna()

    global_mean = np.average(
        values.loc[valid],
        weights=weights[valid.to_numpy()],
    )

    estimate = np.full(
        len(hospital),
        global_mean,
        dtype=float,
    )

    support = np.zeros(
        len(hospital), dtype=float
    )

    used_level = np.zeros(
        len(hospital), dtype=int
    )

    for level, keys in enumerate(
        LEVELS, start=1
    ):
        grouped = group_statistics(
            meps, keys, source, weights
        )

        old_index = hospital.index

        matched = hospital[keys].merge(
            grouped[
                keys + [
                    "mean", "weighted_n", "n"
                ]
            ],
            on=keys,
            how="left",
            sort=False,
            validate="many_to_one",
        )

        matched.index = old_index

        n = matched["n"].fillna(0).to_numpy()
        effective_n = (
            matched["weighted_n"]
            .fillna(0)
            .to_numpy()
        )

        group_mean = (
            matched["mean"]
            .to_numpy(dtype=float)
        )

        acceptable = (
            (n >= min_cell)
            & np.isfinite(group_mean)
        )

        shrink = (
            effective_n
            / (effective_n + alpha)
        )

        new_estimate = (
            shrink * np.nan_to_num(
                group_mean, nan=0.0
            )
            + (1 - shrink) * estimate
        )

        estimate = np.where(
            acceptable,
            new_estimate,
            estimate,
        )

        support = np.where(
            acceptable,
            effective_n,
            support,
        )

        used_level = np.where(
            acceptable,
            level,
            used_level,
        )

    return (
        estimate.astype("float32"),
        support.astype("float32"),
        used_level,
    )


def add_external_statistics(
    meps, hospital, min_cell, alpha
):
    df = hospital.copy()

    print(
        "\n========== CELL STATISTICS ==========",
        flush=True,
    )

    for name, source in METRICS.items():
        est, support, level = attach_cell_stat(
            meps,
            df,
            source,
            min_cell,
            alpha,
        )

        df[name] = est

        # Only report support/level; they are not
        # model features in this experiment.
        dist = {
            int(k): int(v)
            for k, v in zip(
                *np.unique(
                    level, return_counts=True
                )
            )
        }

        print(
            f"[Cell] {name:28s} "
            f"mean={np.mean(est):.4f} "
            f"std={np.std(est):.4f} "
            f"level_counts={dist} "
            f"support_mean={support.mean():.1f}",
            flush=True,
        )

    return df


# ============================================================
# Hospital model and preprocessing
# ============================================================

def new_model(seed):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3,
        scale_pos_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
    )


def feature_columns(df, clinical, experiment):
    generated = set(
        clinical + DIRECT_COLS + ALL_STATS
    )

    excluded = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target",
        "cell_age",
        "cell_sex",
        "cell_race",
    } | generated

    # Extra bridge columns from load_hospital are
    # intentionally excluded from the baseline.
    original = [
        c for c in df.columns
        if (
            c not in excluded
            and not c.startswith("bridge_")
            and not c.startswith("meps_")
            and not c.startswith("direct_")
        )
    ]

    return (
        original
        + clinical
        + EXPERIMENTS[experiment]
    )


def encode(train, val, test, cols):
    matrices = make_features(
        train, val, test, cols
    )

    return [
        x.replace(
            [np.inf, -np.inf], np.nan
        ).astype("float32")
        for x in matrices
    ]


def find_threshold(y, pred):
    ts = np.arange(
        0.05, 0.6001, 0.005
    )

    scores = [
        f1_score(
            y,
            pred >= t,
            zero_division=0,
        )
        for t in ts
    ]

    return float(
        ts[np.argmax(scores)]
    )


# ============================================================
# 5-fold patient-disjoint CV
# ============================================================

def evaluate(df, clinical, folds, seed, outdir):
    results = []
    all_oof = []

    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    splits = cv.split(
        df,
        df["target"],
        groups=df["patient_nbr"],
    )

    for fold, (tr, te) in enumerate(
        splits, start=1
    ):
        train = df.iloc[tr].copy()
        test = df.iloc[te].copy()

        assert set(
            train.patient_nbr
        ).isdisjoint(
            set(test.patient_nbr)
        )

        inner_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=seed + fold,
        )

        it, iv = next(
            inner_split.split(
                train,
                groups=train.patient_nbr,
            )
        )

        inner = train.iloc[it]
        val = train.iloc[iv]

        for part in [
            train, inner, val, test
        ]:
            if part.target.nunique() != 2:
                raise ValueError(
                    "Single-class Hospital split"
                )

        oof = test[
            [
                "encounter_id",
                "patient_nbr",
                "target",
            ]
        ].copy()

        oof = oof.rename(
            columns={"target": "y_true"}
        )

        oof["Fold"] = fold

        print(
            f"\n========== HOSPITAL FOLD "
            f"{fold}/{folds} ==========",
            flush=True,
        )

        for experiment in EXPERIMENTS:
            cols = feature_columns(
                df, clinical, experiment
            )

            # Threshold selection uses only
            # the outer training fold.
            xi, xv, _ = encode(
                inner, val, val, cols
            )

            temp = new_model(seed + fold)
            temp.fit(
                xi, inner.target
            )

            pv = temp.predict_proba(
                xv
            )[:, 1]

            threshold = find_threshold(
                val.target.to_numpy(),
                pv,
            )

            # Final model trained on the entire
            # outer training fold.
            xt, _, xh = encode(
                train, train, test, cols
            )

            model = new_model(seed + fold)

            model.fit(
                xt, train.target
            )

            pred = model.predict_proba(
                xh
            )[:, 1]

            y = test.target.to_numpy()

            row = {
                "Fold": fold,
                "Experiment": experiment,
                "Features": xt.shape[1],
                "AP": average_precision_score(
                    y, pred
                ),
                "AUROC": roc_auc_score(
                    y, pred
                ),
                "F1": f1_score(
                    y,
                    pred >= threshold,
                    zero_division=0,
                ),
                "Brier": brier_score_loss(
                    y, pred
                ),
                "Threshold": threshold,
            }

            results.append(row)

            oof["p_" + experiment] = pred

            print(
                f"[Done] Fold={fold} "
                f"{experiment:24s} "
                f"AP={row['AP']:.5f} "
                f"AUROC={row['AUROC']:.5f} "
                f"F1={row['F1']:.5f}",
                flush=True,
            )

        all_oof.append(oof)

        pd.DataFrame(results).to_csv(
            outdir / "partial_results.csv",
            index=False,
        )

    return (
        pd.DataFrame(results),
        pd.concat(
            all_oof, ignore_index=True
        ),
    )


# ============================================================
# Paired bootstrap
# ============================================================

def bootstrap_comparisons(
    oof, repetitions, seed, outdir
):
    patients, inverse = np.unique(
        oof.patient_nbr.to_numpy(),
        return_inverse=True,
    )

    rng = np.random.default_rng(
        seed + 9000
    )

    y = oof.y_true.to_numpy()

    comparisons = [
        (exp, "Clinical")
        for exp in EXPERIMENTS
        if exp != "Clinical"
    ]

    comparisons.append(
        (
            "Direct_Demog_Cell_All",
            "Direct_Demog",
        )
    )

    rows = []

    for experimental, reference in comparisons:
        p1 = oof[
            "p_" + experimental
        ].to_numpy()

        p0 = oof[
            "p_" + reference
        ].to_numpy()

        observed = (
            average_precision_score(y, p1)
            - average_precision_score(y, p0)
        )

        deltas = []

        for _ in range(repetitions):
            picked = rng.integers(
                0,
                len(patients),
                size=len(patients),
            )

            counts = np.bincount(
                picked,
                minlength=len(patients),
            )

            weights = counts[inverse]

            d = (
                average_precision_score(
                    y, p1,
                    sample_weight=weights,
                )
                - average_precision_score(
                    y, p0,
                    sample_weight=weights,
                )
            )

            deltas.append(d)

        deltas = np.asarray(deltas)

        rows.append({
            "Comparison":
                experimental + " - " + reference,
            "PooledOOF_DeltaAP": observed,
            "CI95_Lower":
                np.quantile(deltas, 0.025),
            "CI95_Upper":
                np.quantile(deltas, 0.975),
            "FractionPositive":
                np.mean(deltas > 0),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        outdir / "patient_bootstrap.csv",
        index=False,
    )

    print(
        "\n========== PATIENT BOOTSTRAP ==========",
        flush=True,
    )

    print(
        result.round(6).to_string(
            index=False
        ),
        flush=True,
    )


# ============================================================
# Summary
# ============================================================

def summarize(results, oof, outdir):
    results.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    oof.to_csv(
        outdir / "oof_all.csv",
        index=False,
    )

    summary = (
        results.groupby(
            "Experiment", sort=False
        )[["AP", "AUROC", "F1", "Brier"]]
        .agg(["mean", "std"])
    )

    summary.columns = [
        a + "_" + b
        for a, b in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        outdir / "summary.csv",
        index=False,
    )

    print(
        "\n========== FINAL 5-FOLD SUMMARY ==========",
        flush=True,
    )

    print(
        summary.round(6).to_string(
            index=False
        ),
        flush=True,
    )

    pivot = results.pivot(
        index="Fold",
        columns="Experiment",
        values="AP",
    )

    rows = []

    for exp in EXPERIMENTS:
        if exp == "Clinical":
            continue

        d = pivot[exp] - pivot["Clinical"]

        rows.append({
            "Comparison": exp + " - Clinical",
            "MeanDeltaAP": d.mean(),
            "StdDeltaAP": d.std(),
            "PositiveFolds": int(
                (d > 0).sum()
            ),
            "PooledOOF_DeltaAP":
                average_precision_score(
                    oof.y_true,
                    oof["p_" + exp],
                )
                - average_precision_score(
                    oof.y_true,
                    oof.p_Clinical,
                ),
        })

    d = (
        pivot["Direct_Demog_Cell_All"]
        - pivot["Direct_Demog"]
    )

    rows.append({
        "Comparison":
            "Direct_Demog_Cell_All - Direct_Demog",
        "MeanDeltaAP": d.mean(),
        "StdDeltaAP": d.std(),
        "PositiveFolds": int(
            (d > 0).sum()
        ),
        "PooledOOF_DeltaAP":
            average_precision_score(
                oof.y_true,
                oof.p_Direct_Demog_Cell_All,
            )
            - average_precision_score(
                oof.y_true,
                oof.p_Direct_Demog,
            ),
    })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        outdir / "paired_comparisons.csv",
        index=False,
    )

    print(
        "\n========== PAIRED COMPARISONS ==========",
        flush=True,
    )

    print(
        comparison.round(6).to_string(
            index=False
        ),
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():
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
        "--outdir",
        default=result_dir("meps_cell_results"),
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
        "--min-cell",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=50,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if (
        args.folds < 2
        or args.min_cell < 1
        or args.alpha < 0
        or args.bootstrap < 0
    ):
        parser.error(
            "Invalid folds/min-cell/alpha/bootstrap"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========== LOAD DATA ==========",
        flush=True,
    )

    hospital, clinical = load_uci(
        args.hospital
    )

    meps = load_meps(
        args.meps
    )

    hospital = add_external_statistics(
        meps,
        hospital,
        args.min_cell,
        args.alpha,
    )

    results, oof = evaluate(
        hospital,
        clinical,
        args.folds,
        args.seed,
        outdir,
    )

    summarize(
        results, oof, outdir
    )

    if args.bootstrap > 0:
        bootstrap_comparisons(
            oof,
            args.bootstrap,
            args.seed,
            outdir,
        )

    print(
        "\n========== FINISHED ==========",
        flush=True,
    )

    print(
        "Saved:", outdir.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()
