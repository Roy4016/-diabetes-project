
import argparse
import json
import re
from pathlib import Path
from _paths import CDC_RAW, UCI_RAW, result_dir

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


SEED = 777

BRIDGE_COLS = [
    "bridge_age",
    "bridge_sex",
    "bridge_high_bp",
    "bridge_high_chol",
]

EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# CDC Age category -> approximate age midpoint.
# Category 13 is 80+, so 85 is an assumption.
CDC_AGE_MID = {
    1: 21, 2: 27, 3: 32, 4: 37,
    5: 42, 6: 47, 7: 52, 8: 57,
    9: 62, 10: 67, 11: 72, 12: 77,
    13: 85,
}


def load_cdc(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [
        "Age", "Sex", "HighBP", "HighChol",
        "GenHlth", "Diabetes_012",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CDC missing columns: {missing}\n"
            "Use the diabetes_012 health indicators CSV."
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Common bridge features
    df["bridge_age"] = df["Age"].map(CDC_AGE_MID)
    df["bridge_sex"] = df["Sex"]
    df["bridge_high_bp"] = df["HighBP"]
    df["bridge_high_chol"] = df["HighChol"]

    # GenHlth: 1=Excellent, 5=Poor.
    # Fair (4) and Poor (5) -> positive class.
    df = df.dropna(subset=BRIDGE_COLS + ["GenHlth"])

    for c in ["bridge_sex", "bridge_high_bp",
              "bridge_high_chol"]:
        df = df[df[c].isin([0, 1])]

    df = df[df["GenHlth"].isin([1, 2, 3, 4, 5])]
    df["poor_health"] = (df["GenHlth"] >= 4).astype(int)

    print(f"[CDC] Total valid samples: {len(df):,}")
    print(
        "[CDC] Poor health prevalence:",
        round(df["poor_health"].mean(), 4)
    )
    return df


def hospital_age_mid(value):
    match = re.match(
        r"^\[(\d+)-(\d+)\)$",
        str(value).strip()
    )
    if not match:
        return np.nan
    lo, hi = map(int, match.groups())
    return (lo + hi) / 2


def icd_has_code(df, start, end):
    """
    Detect ICD-9 numeric prefixes in diag_1, diag_2, diag_3.
    This is only a recorded-diagnosis proxy, not true disease
    prevalence or a complete comorbidity history.
    """
    result = pd.Series(False, index=df.index)

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col not in df.columns:
            raise ValueError(f"Missing diagnosis column: {col}")

        text = df[col].astype("string").str.strip()
        prefix = pd.to_numeric(
            text.str.extract(r"^(\d{3})(?:\.|$)", expand=False),
            errors="coerce",
        )
        result |= prefix.between(start, end).fillna(False)

    return result.astype(np.int8)


def load_hospital(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    required = [
        "patient_nbr", "readmitted", "age", "gender",
        "diag_1", "diag_2", "diag_3",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Hospital missing columns: {missing}"
        )

    df = df.replace("?", np.nan)
    df = df.dropna(subset=["patient_nbr", "readmitted"])

    df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()

    # Match previous preprocessing where discharge outcomes
    # involving death or hospice are excluded.
    if "discharge_disposition_id" in df.columns:
        discharge = pd.to_numeric(
            df["discharge_disposition_id"],
            errors="coerce",
        )
        df = df[
            ~discharge.isin(EXCLUDE_DISCHARGE_IDS)
        ].copy()

    # Binary readmission target
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # Align age and sex
    df["bridge_age"] = df["age"].map(hospital_age_mid)

    df["bridge_sex"] = (
        df["gender"].astype("string").str.strip().str.lower()
        .map({"female": 0, "male": 1})
    )

    # Recorded diagnosis proxies
    df["bridge_high_bp"] = icd_has_code(df, 401, 405)
    df["bridge_high_chol"] = icd_has_code(df, 272, 272)

    df = df.reset_index(drop=True)

    print(f"[Hospital] Samples: {len(df):,}")
    print(
        "[Hospital] Unique patients:",
        df["patient_nbr"].nunique()
    )
    print(
        "[Hospital] Readmission prevalence:",
        round(df["target"].mean(), 4)
    )

    return df


def train_cdc_proxy(cdc, diabetes_only=False):
    if diabetes_only:
        data = cdc[cdc["Diabetes_012"] == 2].copy()
        label = "CDC Diabetes Only"
    else:
        data = cdc.copy()
        label = "CDC All"

    if len(data) == 0 or data["poor_health"].nunique() < 2:
        raise ValueError(
            f"{label}: insufficient samples/classes."
        )

    X = data[BRIDGE_COLS].astype(float)
    y = data["poor_health"]

    neg = (y == 0).sum()
    pos = (y == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=1.0,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device = "cuda",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=neg / max(pos, 1),
    )
    model.fit(X, y)

    print(
        f"[Proxy] {label}: {len(data):,} samples, "
        f"poor health={y.mean():.4f}"
    )
    return model


def patient_disjoint_split(df):
    groups = df["patient_nbr"]

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED,
    )
    train_idx, temp_idx = next(
        first.split(df, df["target"], groups)
    )

    temp_df = df.iloc[temp_idx]

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED + 1,
    )
    val_local, test_local = next(
        second.split(
            temp_df,
            temp_df["target"],
            temp_df["patient_nbr"],
        )
    )

    val_idx = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    sets = [
        set(x["patient_nbr"]) for x in (train, val, test)
    ]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])

    for name, part in [
        ("Train", train), ("Val", val), ("Test", test)
    ]:
        print(
            f"[Split] {name}: {len(part):,} rows, "
            f"positive={part['target'].mean():.4f}"
        )
        if part["target"].nunique() < 2:
            raise ValueError(
                f"{name} contains only one target class."
            )

    return train, val, test


def feature_columns(df, experiment):
    excluded = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target",
        *BRIDGE_COLS,
        "cdc_proxy_all",
        "cdc_proxy_diabetes",
    }

    cols = [
        c for c in df.columns if c not in excluded
    ]

    if experiment != "E1_Baseline":
        cols += BRIDGE_COLS

    if experiment == "E3_CDC_All":
        cols += ["cdc_proxy_all"]

    if experiment == "E4_CDC_Diabetes":
        cols += ["cdc_proxy_diabetes"]

    return cols


def make_features(train, val, test, cols):
    """
    Fit categorical vocabulary on training data only.
    Unknown categories in val/test are ignored.
    """
    xtr = train[cols].copy()
    xva = val[cols].copy()
    xte = test[cols].copy()

    categorical = []

    for c in cols:
        if c in BRIDGE_COLS or c.startswith("cdc_proxy_"):
            for frame in (xtr, xva, xte):
                frame[c] = pd.to_numeric(
                    frame[c], errors="coerce"
                )
        elif (
            pd.api.types.is_object_dtype(xtr[c])
            or isinstance(xtr[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(xtr[c])
            or c in [
                "admission_type_id",
                "admission_source_id",
                "discharge_disposition_id",
            ]
        ):
            categorical.append(c)
            for frame in (xtr, xva, xte):
                frame[c] = (
                    frame[c].fillna("__MISSING__").astype(str)
                )
        else:
            for frame in (xtr, xva, xte):
                frame[c] = pd.to_numeric(
                    frame[c], errors="coerce"
                )

    xtr = pd.get_dummies(
        xtr, columns=categorical, dtype=np.uint8
    )
    xva = pd.get_dummies(
        xva, columns=categorical, dtype=np.uint8
    )
    xte = pd.get_dummies(
        xte, columns=categorical, dtype=np.uint8
    )

    # Training vocabulary only
    xva = xva.reindex(columns=xtr.columns, fill_value=0)
    xte = xte.reindex(columns=xtr.columns, fill_value=0)

    # XGBoost feature names cannot contain [, ] or <
    # Build a unique, stable feature name for every column.
    def sanitize_name(name):
        name = str(name)
        name = re.sub(r"[\[\]<>]", "_", name)
        name = re.sub(r"\s+", "_", name)
        return name

    safe_names = []
    used_names = set()

    for i, name in enumerate(xtr.columns):
        safe_name = f"f{i:04d}_{sanitize_name(name)}"

        if safe_name in used_names:
            raise ValueError(f"Duplicate feature name: {safe_name}")

        used_names.add(safe_name)
        safe_names.append(safe_name)

    xtr.columns = safe_names
    xva.columns = safe_names
    xte.columns = safe_names

    return xtr, xva, xte


def choose_f1_threshold(y_true, probability):
    thresholds = np.arange(0.05, 0.601, 0.005)

    scores = [
        f1_score(
            y_true,
            (probability >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    return float(thresholds[np.argmax(scores)])


def evaluate(y, probability, threshold):
    pred = (probability >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y, probability),
        "PR_AUC_AP": average_precision_score(y, probability),
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(confusion_matrix(y, pred)[0, 0]),
        "FP": int(confusion_matrix(y, pred)[0, 1]),
        "FN": int(confusion_matrix(y, pred)[1, 0]),
        "TP": int(confusion_matrix(y, pred)[1, 1]),
    }


def run_experiment(name, train, val, test, outdir):
    cols = feature_columns(train, name)

    xtr, xva, xte = make_features(
        train, val, test, cols
    )

    ytr = train["target"]
    yva = val["target"]
    yte = test["target"]

    neg = (ytr == 0).sum()
    pos = (ytr == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device = "cuda", 
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=neg / max(pos, 1),
    )

    model.fit(xtr, ytr)

    val_prob = model.predict_proba(xva)[:, 1]
    threshold = choose_f1_threshold(yva, val_prob)
    val_metrics = evaluate(yva, val_prob, threshold)

    test_prob = model.predict_proba(xte)[:, 1]
    metrics = evaluate(yte, test_prob, threshold)

    result = {
        "Experiment": name,
        "Features": xtr.shape[1],
        "Threshold": threshold,
        "Val_PR_AUC_AP": val_metrics["PR_AUC_AP"],
        "Val_F1": val_metrics["F1"],
        **metrics,
    }

    pd.DataFrame({
        "patient_nbr": test["patient_nbr"].values,
        "y_true": yte.values,
        "y_prob": test_prob,
        "y_pred": (test_prob >= threshold).astype(int),
    }).to_csv(outdir / f"{name}_predictions.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "feature_columns": xtr.columns.tolist(),
            "threshold": threshold,
        },
        outdir / f"{name}_model.joblib",
    )

    cm = confusion_matrix(
        yte, (test_prob >= threshold).astype(int)
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        cm, display_labels=["Not <30", "<30"]
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(
        outdir / f"{name}_confusion_matrix.png",
        dpi=160
    )
    plt.close(fig)

    print(
        f"[Done] {name}: "
        f"threshold={threshold:.3f}, "
        f"AUROC={metrics['AUROC']:.4f}, "
        f"AP={metrics['PR_AUC_AP']:.4f}, "
        f"F1={metrics['F1']:.4f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital",
        default=UCI_RAW,
    )
    parser.add_argument(
        "--cdc",
        default=CDC_RAW,
    )
    parser.add_argument(
        "--outdir",
        default=result_dir("cdc_proxy_xgb_results"),
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== LOAD DATA ==========")
    hospital = load_hospital(args.hospital)
    cdc = load_cdc(args.cdc)

    print("\n========== TRAIN CDC PROXIES ==========")
    proxy_all_path = outdir / "cdc_proxy_all.joblib"
    proxy_diabetes_path = outdir / "cdc_proxy_diabetes.joblib"

    if proxy_all_path.exists() and proxy_diabetes_path.exists():
        print("[Proxy] Loading existing CDC models...")

        proxy_all = joblib.load(proxy_all_path)
        proxy_diabetes = joblib.load(proxy_diabetes_path)

    else:
        print("[Proxy] Training CDC models...")

        proxy_all = train_cdc_proxy(
            cdc, diabetes_only=False
        )
        proxy_diabetes = train_cdc_proxy(
            cdc, diabetes_only=True
        )

        joblib.dump(proxy_all, proxy_all_path)
        joblib.dump(proxy_diabetes, proxy_diabetes_path)

    hospital_bridge = hospital[BRIDGE_COLS].astype(float)

    hospital["cdc_proxy_all"] = (
        proxy_all.predict_proba(hospital_bridge)[:, 1]
    )
    hospital["cdc_proxy_diabetes"] = (
        proxy_diabetes.predict_proba(hospital_bridge)[:, 1]
    )

    joblib.dump(proxy_all, outdir / "cdc_proxy_all.joblib")
    joblib.dump(
        proxy_diabetes,
        outdir / "cdc_proxy_diabetes.joblib"
    )

    print("\n========== PATIENT SPLIT ==========")
    train, val, test = patient_disjoint_split(hospital)

    # Save shared split assignments
    split_records = []
    for label, part in [
        ("train", train), ("val", val), ("test", test)
    ]:
        temp = part[["patient_nbr"]].copy()
        temp["split"] = label
        if "encounter_id" in part.columns:
            temp["encounter_id"] = part["encounter_id"]
        split_records.append(temp)

    pd.concat(split_records).to_csv(
        outdir / "split_assignments.csv",
        index=False,
    )

    experiments = [
        "E1_Baseline",
        "E2_Bridge",
        "E3_CDC_All",
        "E4_CDC_Diabetes",
    ]

    print("\n========== XGBOOST EXPERIMENTS ==========")
    results = []

    for name in experiments:
        results.append(
            run_experiment(
                name, train, val, test, outdir
            )
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        outdir / "experiment_results.csv",
        index=False,
    )

    print("\n========== FINAL RESULTS ==========")
    print(
        result_df[
            [
                "Experiment", "AUROC", "PR_AUC_AP",
                "Precision", "Recall", "F1",
                "Threshold"
            ]
        ].round(4).to_string(index=False)
    )

    baseline_ap = result_df.iloc[0]["PR_AUC_AP"]

    print("\n========== VS BASELINE ==========")
    for _, row in result_df.iterrows():
        delta = row["PR_AUC_AP"] - baseline_ap
        print(
            f"{row['Experiment']}: "
            f"delta AP={delta:+.4f}"
        )

    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
