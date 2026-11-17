
import argparse
import json
import re
from pathlib import Path
from _paths import CDC_RAW, UCI_RAW, result_dir

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import run_cdc_xgb as old
from run_cdc_xgb import (
    BRIDGE_COLS,
    load_hospital,
    load_cdc,
    make_features,
    choose_f1_threshold,
)


PROXY_TARGETS = {
    "proxy_poor_health": "GenHlth",
    "proxy_diff_walk": "DiffWalk",
    "proxy_inactive": "PhysActivity",
    "proxy_smoker": "Smoker",
}

FEATURE_SETS = [
    "Baseline",
    "Bridge",
    "CDC_All",
    "CDC_Multi",
]

IH_METHODS = [
    "None",
    "ClassWeight",
    "ROS",
    "RUS",
    "SMOTE",
]

ALL_NEW_PROXIES = list(PROXY_TARGETS)

EXCLUDE_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "target",
    "cdc_proxy_all",
    "cdc_proxy_diabetes",
    *BRIDGE_COLS,
    *ALL_NEW_PROXIES,
}


def create_model(seed, scale_pos_weight=1.0):
    return XGBClassifier(
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
        device="cuda",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def load_cdc_targets(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [
        "Age", "Sex", "HighBP", "HighChol",
        "Diabetes_012",
        *PROXY_TARGETS.values(),
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CDC columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df["bridge_age"] = df["Age"].map(
        old.CDC_AGE_MID
    )
    df["bridge_sex"] = df["Sex"]
    df["bridge_high_bp"] = df["HighBP"]
    df["bridge_high_chol"] = df["HighChol"]

    df = df.dropna(subset=BRIDGE_COLS)

    for col in BRIDGE_COLS[1:]:
        df = df[df[col].isin([0, 1])]

    return df


def train_cdc_targets(cdc, hospital, seed):
    print("\n========== TRAIN CDC TARGETS ==========")

    # Keep the original all-population proxy
    # comparable with your previous experiments.
    old.SEED = seed

    cdc_original = cdc.dropna(
        subset=["GenHlth"]
    ).copy()

    cdc_original = cdc_original[
        cdc_original["GenHlth"].isin([1, 2, 3, 4, 5])
    ]

    cdc_original["poor_health"] = (
        cdc_original["GenHlth"] >= 4
    ).astype(int)

    proxy_all = old.train_cdc_proxy(
        cdc_original, diabetes_only=False
    )

    X_hospital = hospital[BRIDGE_COLS].astype(float)

    hospital["cdc_proxy_all"] = (
        proxy_all.predict_proba(X_hospital)[:, 1]
    )

    diabetes = cdc[
        cdc["Diabetes_012"] == 2
    ].copy()

    print(
        f"[CDC Diabetes] {len(diabetes):,} samples"
    )

    for proxy_name, target_col in PROXY_TARGETS.items():
        data = diabetes.dropna(
            subset=[target_col]
        ).copy()

        if proxy_name == "proxy_poor_health":
            data = data[
                data[target_col].isin([1, 2, 3, 4, 5])
            ]
            y = (data[target_col] >= 4).astype(int)

        elif proxy_name == "proxy_inactive":
            data = data[
                data[target_col].isin([0, 1])
            ]
            y = (data[target_col] == 0).astype(int)

        else:
            data = data[
                data[target_col].isin([0, 1])
            ]
            y = (data[target_col] == 1).astype(int)

        if y.nunique() != 2:
            raise ValueError(
                f"Invalid CDC target: {proxy_name}"
            )

        X = data[BRIDGE_COLS].astype(float)

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
            device="cuda",
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(X, y)

        hospital[proxy_name] = (
            model.predict_proba(X_hospital)[:, 1]
        )

        print(
            f"[Proxy] {proxy_name}: "
            f"n={len(data):,}, "
            f"positive={y.mean():.4f}, "
            f"hospital_std="
            f"{hospital[proxy_name].std():.5f}",
            flush=True,
        )

    return hospital


def get_feature_columns(df, feature_set):
    # A strict exclusion list prevents proxy features
    # from accidentally entering the control groups.
    cols = [
        col for col in df.columns
        if col not in EXCLUDE_COLUMNS
    ]

    if feature_set in [
        "Bridge", "CDC_All", "CDC_Multi"
    ]:
        cols += BRIDGE_COLS

    if feature_set == "CDC_All":
        cols += ["cdc_proxy_all"]

    elif feature_set == "CDC_Multi":
        cols += ALL_NEW_PROXIES

    return list(dict.fromkeys(cols))


def encode_and_impute(train, val, test, cols):
    # One-hot encoding vocabulary is learned
    # only from the training subset.
    xtr, xva, xte = make_features(
        train, val, test, cols
    )

    # SMOTE does not support NaN.
    # For comparison fairness, apply identical
    # missing-value handling to every IH method.
    xtr = xtr.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    xva = xva.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    xte = xte.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    # Train-only median imputation.
    medians = xtr.median().fillna(0)

    xtr = xtr.fillna(medians).fillna(0)
    xva = xva.fillna(medians).fillna(0)
    xte = xte.fillna(medians).fillna(0)

    return xtr, xva, xte


def apply_imbalance_method(X, y, method, seed, ratio):
    y = np.asarray(y, dtype=np.int32)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    current_ratio = n_pos / max(n_neg, 1)

    if method == "None":
        return X, y, 1.0

    if method == "ClassWeight":
        return X, y, n_neg / max(n_pos, 1)

    if method == "ROS":
        if ratio <= current_ratio:
            return X, y, 1.0

        sampler = RandomOverSampler(
            sampling_strategy=ratio,
            random_state=seed,
        )

    elif method == "RUS":
        if ratio <= current_ratio:
            return X, y, 1.0

        sampler = RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=seed,
        )

    elif method == "SMOTE":
        if ratio <= current_ratio:
            return X, y, 1.0

        # Exploratory only: this interpolates one-hot
        # features, which may create fractional dummies.
        sampler = SMOTE(
            sampling_strategy=ratio,
            random_state=seed,
            k_neighbors=5,
        )

    else:
        raise ValueError(f"Unknown IH method: {method}")

    X_new, y_new = sampler.fit_resample(X, y)

    if not isinstance(X_new, pd.DataFrame):
        X_new = pd.DataFrame(
            X_new, columns=X.columns
        )

    X_new = X_new.astype(np.float32)

    return X_new, np.asarray(y_new), 1.0


def compute_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y, prob),
        "AP": average_precision_score(y, prob),
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "Recall": recall_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "Brier": brier_score_loss(y, prob),
    }


def run_one(
    df,
    outer_train_idx,
    outer_test_idx,
    feature_set,
    method,
    fold,
    seed,
    ratio,
    outdir,
):
    outer_train = df.iloc[
        outer_train_idx
    ].copy()

    outer_test = df.iloc[
        outer_test_idx
    ].copy()

    cols = get_feature_columns(
        df, feature_set
    )

    # Inner patient-group split, only for threshold.
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed + fold,
    )

    itr, iva = next(
        inner_split.split(
            outer_train,
            outer_train["target"],
            outer_train["patient_nbr"],
        )
    )

    inner_train = outer_train.iloc[itr].copy()
    inner_val = outer_train.iloc[iva].copy()

    assert set(
        inner_train["patient_nbr"]
    ).isdisjoint(set(inner_val["patient_nbr"]))

    if (
        inner_train["target"].nunique() != 2
        or inner_val["target"].nunique() != 2
    ):
        raise ValueError(
            f"Inner split fold {fold} missing class."
        )

    xi, xv, _ = encode_and_impute(
        inner_train, inner_val, inner_val, cols
    )

    yi = inner_train["target"].to_numpy()
    yv = inner_val["target"].to_numpy()

    xi, yi, weight = apply_imbalance_method(
        xi, yi, method, seed + fold, ratio
    )

    inner_model = create_model(
        seed + fold, weight
    )
    inner_model.fit(xi, yi)

    val_prob = inner_model.predict_proba(
        xv
    )[:, 1]

    threshold = choose_f1_threshold(
        yv, val_prob
    )

    # Refit on the full outer training fold.
    xt, _, xh = encode_and_impute(
        outer_train, outer_train, outer_test, cols
    )

    yt = outer_train["target"].to_numpy()
    yh = outer_test["target"].to_numpy()

    xt, yt, weight = apply_imbalance_method(
        xt, yt, method, seed + fold, ratio
    )

    model = create_model(
        seed + fold, weight
    )
    model.fit(xt, yt)

    prob = model.predict_proba(xh)[:, 1]

    metrics = compute_metrics(
        yh, prob, threshold
    )

    result = {
        "FeatureSet": feature_set,
        "IH": method,
        "Fold": fold,
        "TrainRows": len(outer_train),
        "TestRows": len(outer_test),
        "PositiveRate": float(yh.mean()),
        "Threshold": float(threshold),
        **metrics,
    }

    print(
        f"[Done] Fold={fold} "
        f"{feature_set:10s} "
        f"{method:12s} "
        f"AP={metrics['AP']:.4f} "
        f"AUROC={metrics['AUROC']:.4f} "
        f"F1={metrics['F1']:.4f}",
        flush=True,
    )

    return result


def summarize(results, outdir):
    df = pd.DataFrame(results)

    df.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    metrics = [
        "AP", "AUROC", "Precision",
        "Recall", "F1", "Brier"
    ]

    summary = (
        df.groupby(["FeatureSet", "IH"])[metrics]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        outdir / "summary_mean_std.csv",
        index=False,
    )

    print("\n========== FINAL SUMMARY ==========")

    print(
        summary[
            [
                "FeatureSet",
                "IH",
                "AP_mean",
                "AP_std",
                "AUROC_mean",
                "F1_mean",
                "Brier_mean",
            ]
        ]
        .sort_values(
            "AP_mean", ascending=False
        )
        .round(4)
        .to_string(index=False)
    )

    # Pair every CDC setting with Bridge under
    # exactly the same fold and IH method.
    bridge = df[
        df["FeatureSet"] == "Bridge"
    ][["Fold", "IH", "AP"]].rename(
        columns={"AP": "Bridge_AP"}
    )

    comparison = df[
        df["FeatureSet"].isin([
            "CDC_All", "CDC_Multi"
        ])
    ].merge(
        bridge,
        on=["Fold", "IH"],
        how="inner",
        validate="many_to_one",
    )

    comparison["Delta_AP"] = (
        comparison["AP"] - comparison["Bridge_AP"]
    )

    delta = (
        comparison.groupby(["FeatureSet", "IH"])
        .agg(
            Mean_Delta_AP=("Delta_AP", "mean"),
            Std_Delta_AP=("Delta_AP", "std"),
            Positive_Folds=(
                "Delta_AP",
                lambda x: int((x > 0).sum())
            ),
        )
        .reset_index()
    )

    delta.to_csv(
        outdir / "paired_delta_vs_bridge.csv",
        index=False,
    )

    print("\n========== CDC VS BRIDGE ==========")
    print(
        delta.round(5).to_string(index=False)
    )

    # Fold-level win/loss relative to plain Hospital
    # Baseline with the SAME imbalance strategy.
    base = df[
        df["FeatureSet"] == "Baseline"
    ][["Fold", "IH", "AP"]].rename(
        columns={"AP": "Baseline_AP"}
    )

    vs_base = df.merge(
        base,
        on=["Fold", "IH"],
        how="inner",
        validate="many_to_one",
    )

    vs_base["Delta_AP"] = (
        vs_base["AP"] - vs_base["Baseline_AP"]
    )

    vs_base.to_csv(
        outdir / "paired_vs_baseline.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital",
        default=UCI_RAW
    )
    parser.add_argument(
        "--cdc",
        default=CDC_RAW
    )
    parser.add_argument(
        "--folds", type=int, default=5
    )
    parser.add_argument(
        "--seed", type=int, default=777
    )
    parser.add_argument(
        "--ratio", type=float, default=0.30
    )
    parser.add_argument(
        "--outdir",
        default=result_dir("cdc_ih_5fold_results")
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_SETS,
        default=FEATURE_SETS,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=IH_METHODS,
        default=IH_METHODS,
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2")

    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be in (0, 1]")

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True, exist_ok=True
    )

    print("\n========== DATA ==========")

    hospital = load_hospital(args.hospital)
    load_cdc(args.cdc)
    cdc = load_cdc_targets(args.cdc)

    hospital = train_cdc_targets(
        cdc, hospital, args.seed
    )

    print("\n========== FOLDS ==========")

    cv = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    folds = list(
        cv.split(
            hospital,
            hospital["target"],
            groups=hospital["patient_nbr"],
        )
    )

    results = []

    config = {
        "seed": args.seed,
        "folds": args.folds,
        "ratio": args.ratio,
        "features": args.features,
        "methods": args.methods,
    }

    with open(
        outdir / "config.json", "w"
    ) as f:
        json.dump(config, f, indent=2)

    for fold, (itr, ite) in enumerate(
        folds, start=1
    ):
        train_ids = set(
            hospital.iloc[itr]["patient_nbr"]
        )
        test_ids = set(
            hospital.iloc[ite]["patient_nbr"]
        )

        assert train_ids.isdisjoint(test_ids)

        print(
            f"\n========== FOLD {fold}/{args.folds} "
            f"Train={len(itr):,} "
            f"Test={len(ite):,} ==========",
            flush=True,
        )

        for features in args.features:
            for method in args.methods:
                row = run_one(
                    hospital,
                    itr, ite,
                    features,
                    method,
                    fold,
                    args.seed,
                    args.ratio,
                    outdir,
                )
                results.append(row)

                pd.DataFrame(results).to_csv(
                    outdir / "partial_results.csv",
                    index=False,
                )

    summarize(results, outdir)

    print("\n========== FINISHED ==========")
    print(f"Saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
