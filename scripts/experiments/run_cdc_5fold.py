
import argparse
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
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from run_cdc_xgb import (
    BRIDGE_COLS,
    load_cdc,
    load_hospital,
    train_cdc_proxy,
    feature_columns,
    make_features,
    choose_f1_threshold,
)

EXPERIMENTS = [
    "E1_Baseline",
    "E2_Bridge",
    "E3_CDC_All",
    "E4_CDC_Diabetes",
]


def create_xgb(seed, y_train):
    positive = int(np.sum(y_train == 1))
    negative = int(np.sum(y_train == 0))

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
        scale_pos_weight=negative / max(positive, 1),
    )


def get_features(df, experiment):
    return feature_columns(df, experiment)


def get_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, probability),
        "PR_AUC_AP": average_precision_score(
            y_true, probability
        ),
        "Precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_true, prediction, zero_division=0
        ),
    }


def train_one_fold(
    df,
    train_idx,
    test_idx,
    experiment,
    fold,
    seed,
    outdir,
):
    outer_train = df.iloc[train_idx].copy()
    outer_test = df.iloc[test_idx].copy()

    cols = get_features(outer_train, experiment)

    # ----------------------------------------
    # Step 1: Inner group split
    # Used only for threshold selection.
    # ----------------------------------------
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed + fold,
    )

    inner_train_idx, inner_val_idx = next(
        inner_split.split(
            outer_train,
            outer_train["target"],
            groups=outer_train["patient_nbr"],
        )
    )

    inner_train = outer_train.iloc[
        inner_train_idx
    ].copy()

    inner_val = outer_train.iloc[
        inner_val_idx
    ].copy()

    # Ensure patient independence
    assert set(inner_train["patient_nbr"]).isdisjoint(
        set(inner_val["patient_nbr"])
    )

    if (
        inner_train["target"].nunique() != 2
        or inner_val["target"].nunique() != 2
    ):
        raise ValueError(
            f"Fold {fold}: inner split missing class."
        )

    x_inner, x_val, _ = make_features(
        inner_train, inner_val, inner_val, cols
    )

    y_inner = inner_train["target"].to_numpy()
    y_val = inner_val["target"].to_numpy()

    threshold_model = create_xgb(
        seed + fold, y_inner
    )
    threshold_model.fit(x_inner, y_inner)

    val_probability = threshold_model.predict_proba(
        x_val
    )[:, 1]

    threshold = choose_f1_threshold(
        y_val, val_probability
    )

    # ----------------------------------------
    # Step 2: Refit on ALL outer training data
    # Outer held-out fold is never used to
    # choose the threshold or fit the model.
    # ----------------------------------------
    x_train, _, x_test = make_features(
        outer_train, outer_train, outer_test, cols
    )

    y_train = outer_train["target"].to_numpy()
    y_test = outer_test["target"].to_numpy()

    model = create_xgb(seed + fold, y_train)
    model.fit(x_train, y_train)

    probability = model.predict_proba(
        x_test
    )[:, 1]

    metrics = get_metrics(
        y_test, probability, threshold
    )

    result = {
        "Experiment": experiment,
        "Fold": fold,
        "Seed": seed,
        "Train_Rows": len(outer_train),
        "Test_Rows": len(outer_test),
        "Test_Positive_Rate": float(y_test.mean()),
        "Features": x_train.shape[1],
        "Threshold": threshold,
        **metrics,
    }

    predictions = pd.DataFrame({
        "encounter_id": (
            outer_test["encounter_id"].values
            if "encounter_id" in outer_test
            else outer_test.index.to_numpy()
        ),
        "patient_nbr": outer_test[
            "patient_nbr"
        ].values,
        "y_true": y_test,
        "y_probability": probability,
        "y_pred": (
            probability >= threshold
        ).astype(int),
    })

    predictions.to_csv(
        outdir / f"{experiment}_fold{fold}_pred.csv",
        index=False,
    )

    print(
        f"[Done] {experiment} Fold {fold}: "
        f"AUROC={metrics['AUROC']:.4f}, "
        f"AP={metrics['PR_AUC_AP']:.4f}, "
        f"F1={metrics['F1']:.4f}, "
        f"threshold={threshold:.3f}",
        flush=True,
    )

    return result


def attach_cdc_proxy(hospital, cdc, seed):
    # The original helper uses its module-level SEED.
    # Set it to the chosen experiment seed.
    import run_cdc_xgb
    run_cdc_xgb.SEED = seed

    print("\n========== TRAIN CDC MODELS ==========")

    proxy_all = train_cdc_proxy(
        cdc, diabetes_only=False
    )

    proxy_diabetes = train_cdc_proxy(
        cdc, diabetes_only=True
    )

    bridge = hospital[BRIDGE_COLS].astype(float)

    hospital["cdc_proxy_all"] = (
        proxy_all.predict_proba(bridge)[:, 1]
    )

    hospital["cdc_proxy_diabetes"] = (
        proxy_diabetes.predict_proba(bridge)[:, 1]
    )

    return hospital


def generate_summary(results, outdir):
    df = pd.DataFrame(results)

    df.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    metrics = [
        "AUROC",
        "PR_AUC_AP",
        "Precision",
        "Recall",
        "F1",
    ]

    summary = df.groupby(
        "Experiment", sort=False
    )[metrics].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        outdir / "summary_mean_std.csv",
        index=False,
    )

    # Paired fold-level difference versus E1
    pivot = df.pivot(
        index="Fold",
        columns="Experiment",
        values="PR_AUC_AP",
    )

    delta_rows = []

    for experiment in EXPERIMENTS[1:]:
        delta = (
            pivot[experiment]
            - pivot["E1_Baseline"]
        )

        delta_rows.append({
            "Experiment": experiment,
            "Mean_Delta_AP_vs_E1": delta.mean(),
            "Std_Delta_AP_vs_E1": delta.std(),
            "Positive_Folds": int((delta > 0).sum()),
            "Total_Folds": len(delta),
        })

    # Also compare proxies directly against Bridge
    for experiment in [
        "E3_CDC_All",
        "E4_CDC_Diabetes",
    ]:
        delta = (
            pivot[experiment]
            - pivot["E2_Bridge"]
        )

        delta_rows.append({
            "Experiment": f"{experiment}_vs_E2",
            "Mean_Delta_AP_vs_E1": delta.mean(),
            "Std_Delta_AP_vs_E1": delta.std(),
            "Positive_Folds": int((delta > 0).sum()),
            "Total_Folds": len(delta),
        })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(
        outdir / "paired_delta_ap.csv",
        index=False,
    )

    print("\n========== 5-FOLD SUMMARY ==========")

    display_cols = [
        "Experiment",
        "AUROC_mean",
        "AUROC_std",
        "PR_AUC_AP_mean",
        "PR_AUC_AP_std",
        "F1_mean",
        "F1_std",
    ]

    print(
        summary[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\n========== PAIRED AP DIFFERENCE ==========")

    print(
        delta_df.round(4).to_string(index=False)
    )

    return summary


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
        "--outdir",
        default=result_dir("cdc_5fold_results"),
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== LOAD DATA ==========")

    hospital = load_hospital(args.hospital)
    cdc = load_cdc(args.cdc)

    hospital = attach_cdc_proxy(
        hospital, cdc, args.seed
    )

    print("\n========== CREATE GROUPED FOLDS ==========")

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

    fold_assignments = []

    for fold, (train_idx, test_idx) in enumerate(
        folds, start=1
    ):
        train_patients = set(
            hospital.iloc[train_idx]["patient_nbr"]
        )
        test_patients = set(
            hospital.iloc[test_idx]["patient_nbr"]
        )

        assert train_patients.isdisjoint(test_patients)

        train_rate = hospital.iloc[
            train_idx
        ]["target"].mean()

        test_rate = hospital.iloc[
            test_idx
        ]["target"].mean()

        if (
            hospital.iloc[test_idx]["target"].nunique() != 2
            or hospital.iloc[train_idx]["target"].nunique() != 2
        ):
            raise ValueError(
                f"Outer fold {fold} missing a class."
            )

        print(
            f"Fold {fold}: "
            f"Train={len(train_idx):,}, "
            f"Test={len(test_idx):,}, "
            f"Train Positive={train_rate:.4f}, "
            f"Test Positive={test_rate:.4f}"
        )

        fold_part = hospital.iloc[test_idx][
            ["patient_nbr"]
        ].copy()

        if "encounter_id" in hospital:
            fold_part["encounter_id"] = (
                hospital.iloc[test_idx][
                    "encounter_id"
                ].values
            )

        fold_part["fold"] = fold
        fold_assignments.append(fold_part)

    pd.concat(fold_assignments).to_csv(
        outdir / "fold_assignments.csv",
        index=False,
    )

    results = []

    print("\n========== RUN CROSS VALIDATION ==========")

    for fold, (train_idx, test_idx) in enumerate(
        folds, start=1
    ):
        print(
            f"\n========== FOLD {fold}/{args.folds} ==========",
            flush=True,
        )

        for experiment in EXPERIMENTS:
            result = train_one_fold(
                hospital,
                train_idx,
                test_idx,
                experiment,
                fold,
                args.seed,
                outdir,
            )

            results.append(result)

            # Save partial results for long experiments
            pd.DataFrame(results).to_csv(
                outdir / "fold_results_partial.csv",
                index=False,
            )

    generate_summary(results, outdir)

    print("\n========== FINISHED ==========")
    print(f"Results saved: {outdir.resolve()}")


if __name__ == "__main__":
    main()
