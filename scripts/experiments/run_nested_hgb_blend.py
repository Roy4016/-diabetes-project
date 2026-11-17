#!/usr/bin/env python3
"""Nested HGB tuning and HGB/XGBoost probability blending.

Hyperparameters and blend weights are selected on patient-disjoint inner
validation data. Outer test folds are used only for final evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from _paths import MEPS_BRIDGE, UCI_RAW, result_dir
from run_cdc_encoder_control import external_meps, load_meps, prepare_hospital
from run_cdc_xgb import make_features as make_xgb_features
from run_meps_hgb import raw_discharge_predictors


HGB_GRID = [
    {"learning_rate": .05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"learning_rate": .05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"learning_rate": .05, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": .03, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"learning_rate": .08, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 5.0},
]
BLEND_WEIGHTS = (0.0, .25, .5, .75, 1.0)  # weight assigned to XGBoost


def feature_columns(frame, use_meps):
    columns = raw_discharge_predictors(frame)
    if use_meps:
        columns += ["meps_er_8", "meps_ip_8"]
    return columns


def encode_frames(train, validation, test, columns, min_frequency):
    categorical_ids = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    categorical, numeric = [], []
    normalized = []
    for source in (train, validation, test):
        frame = source.loc[:, columns].copy()
        for name in columns:
            if name in categorical_ids or pd.api.types.is_object_dtype(frame[name]) or pd.api.types.is_string_dtype(frame[name]):
                if name not in categorical:
                    categorical.append(name)
                frame[name] = frame[name].astype(object).where(pd.notna(frame[name]), np.nan)
            else:
                if name not in numeric:
                    numeric.append(name)
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
        normalized.append(frame)
    transformer = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                      sparse_output=False, dtype=np.float32)),
        ]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ], sparse_threshold=0.)
    train_x = transformer.fit_transform(normalized[0]).astype(np.float32, copy=False)
    val_x = transformer.transform(normalized[1]).astype(np.float32, copy=False)
    test_x = transformer.transform(normalized[2]).astype(np.float32, copy=False)
    return train_x, val_x, test_x


def hgb_model(seed, params, early_stopping=True):
    return HistGradientBoostingClassifier(
        **params, early_stopping=early_stopping, validation_fraction=.1,
        n_iter_no_change=20, random_state=seed,
    )


def xgb_model(seed, estimators, params):
    return XGBClassifier(
        n_estimators=estimators, max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], min_child_weight=params["min_child_weight"],
        subsample=.85, colsample_bytree=.85, reg_lambda=params["reg_lambda"],
        scale_pos_weight=params["scale_pos_weight"], objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist", device="cpu", random_state=seed,
        n_jobs=-1,
    )


XGB_GRID = [
    {"max_depth": 3, "learning_rate": .04, "min_child_weight": 5, "reg_lambda": 3, "scale_pos_weight": 1.},
    {"max_depth": 4, "learning_rate": .04, "min_child_weight": 5, "reg_lambda": 3, "scale_pos_weight": 1.},
    {"max_depth": 4, "learning_rate": .03, "min_child_weight": 5, "reg_lambda": 5, "scale_pos_weight": 2.},
    {"max_depth": 5, "learning_rate": .025, "min_child_weight": 8, "reg_lambda": 5, "scale_pos_weight": 2.},
]


def choose_hgb(train_x, val_x, y_train, y_val, seed):
    candidates = []
    for number, params in enumerate(HGB_GRID, 1):
        model = hgb_model(seed + number, params).fit(train_x, y_train)
        probability = model.predict_proba(val_x)[:, 1]
        ap = average_precision_score(y_val, probability)
        candidates.append((ap, params, int(model.n_iter_), probability))
        print(f"  [HGB {number}/{len(HGB_GRID)}] val_PR_AUC={ap:.6f} iter={model.n_iter_} params={params}", flush=True)
    return max(candidates, key=lambda item: item[0])


def choose_xgb(train_x, val_x, y_train, y_val, seed, estimators):
    candidates = []
    for number, params in enumerate(XGB_GRID, 1):
        model = xgb_model(seed + number * 101, estimators, params)
        model.fit(train_x, y_train, verbose=False)
        probability = model.predict_proba(val_x)[:, 1]
        best_iteration = estimators
        ap = average_precision_score(y_val, probability)
        candidates.append((ap, params, best_iteration, probability))
        print(f"  [XGB {number}/{len(XGB_GRID)}] val_PR_AUC={ap:.6f} trees={best_iteration} params={params}", flush=True)
    return max(candidates, key=lambda item: item[0])


def patient_bootstrap(oof, comparisons, repetitions, seed):
    patients, inverse = np.unique(oof.patient_nbr.to_numpy(), return_inverse=True)
    y = oof.y_true.to_numpy()
    rng = np.random.default_rng(seed + 17000)
    rows = []
    for left, right in comparisons:
        p_left, p_right = oof[f"p_{left}"].to_numpy(), oof[f"p_{right}"].to_numpy()
        observed = average_precision_score(y, p_left) - average_precision_score(y, p_right)
        values = []
        for _ in range(repetitions):
            counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
            weights = counts[inverse]
            values.append(average_precision_score(y, p_left, sample_weight=weights)
                          - average_precision_score(y, p_right, sample_weight=weights))
        rows.append({"Comparison": f"{left} - {right}", "Delta_PR_AUC": observed,
                     "CI95_Lower": np.quantile(values, .025), "CI95_Upper": np.quantile(values, .975),
                     "FractionPositive": np.mean(np.asarray(values) > 0), "Bootstraps": repetitions})
    return pd.DataFrame(rows)


def run(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    frame, clinical = prepare_hospital(args.hospital)
    frame = external_meps(load_meps(args.meps), frame, args.seed, out, device="cpu")
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows, oof_parts = [], []
    for fold, (train_ids, test_ids) in enumerate(splitter.split(frame, frame.target, groups=frame.patient_nbr), 1):
        outer_train, test = frame.iloc[train_ids], frame.iloc[test_ids]
        assert set(outer_train.patient_nbr).isdisjoint(set(test.patient_nbr))
        inner_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed + fold)
                                  .split(outer_train, groups=outer_train.patient_nbr))
        inner, val = outer_train.iloc[inner_idx], outer_train.iloc[val_idx]
        if any(part.target.nunique() < 2 for part in (inner, val, outer_train, test)):
            raise ValueError(f"One-class partition in fold {fold}")
        y_inner, y_val = inner.target.to_numpy(), val.target.to_numpy()
        fold_oof = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        fold_oof["Fold"] = fold
        predictions = {}

        # Fit one fold-specific preprocessor for clinical HGB and another for
        # the MEPS-feature XGBoost. All category vocabularies use inner-train only.
        hgb_columns = feature_columns(frame, False)
        x_inner, x_val, _ = encode_frames(inner, val, val, hgb_columns, args.min_frequency)
        x_final_train, _, x_test = encode_frames(outer_train, outer_train, test, hgb_columns, args.min_frequency)
        _, hgb_params, hgb_iter, hgb_val_probability = choose_hgb(
            x_inner, x_val, y_inner, y_val, args.seed + fold,
        )

        xgb_columns = feature_columns(frame, True)
        z_inner, z_val, _ = make_xgb_features(inner, val, val, xgb_columns)
        _, xgb_params, xgb_trees, xgb_val_probability = choose_xgb(
            z_inner, z_val, y_inner, y_val, args.seed + fold, args.xgb_estimators,
        )

        weight_scores = []
        for xgb_weight in args.blend_weights:
            blended = (1 - xgb_weight) * hgb_val_probability + xgb_weight * xgb_val_probability
            weight_scores.append((average_precision_score(y_val, blended), float(xgb_weight)))
        selected_weight = max(weight_scores, key=lambda item: item[0])[1]
        print(f"[Fold {fold}] selected XGB blend weight={selected_weight:.2f} val_PR_AUC={max(weight_scores)[0]:.6f}", flush=True)

        # Fit selected HGB on all outer-train rows and selected XGB on the same
        # rows, then evaluate the locked blend once on the outer test fold.
        final_hgb = hgb_model(
            args.seed + fold, {**hgb_params, "max_iter": hgb_iter}, early_stopping=False,
        ).fit(x_final_train, outer_train.target.to_numpy())
        hgb_test = final_hgb.predict_proba(x_test)[:, 1]
        z_train, _, z_test = make_xgb_features(outer_train, outer_train, test, xgb_columns)
        final_xgb = xgb_model(args.seed + fold, xgb_trees, xgb_params).fit(z_train, outer_train.target.to_numpy(), verbose=False)
        xgb_test = final_xgb.predict_proba(z_test)[:, 1]
        blended_test = (1 - selected_weight) * hgb_test + selected_weight * xgb_test
        predictions = {"hgb_tuned": hgb_test, "xgb_meps_tuned": xgb_test,
                       "ensemble_fixed_50": .5 * hgb_test + .5 * xgb_test,
                       "ensemble_nested": blended_test}
        for name, probability in predictions.items():
            rows.append({"Fold": fold, "Experiment": name,
                         "PR_AUC_AP": average_precision_score(test.target, probability),
                         "AUROC": roc_auc_score(test.target, probability),
                         "Brier": brier_score_loss(test.target, probability),
                         "XGBWeight": selected_weight, "HGBIterations": hgb_iter, "XGBTrees": xgb_trees})
            fold_oof[f"p_{name}"] = probability
        oof_parts.append(fold_oof)
        pd.DataFrame(rows).to_csv(out / "partial_results.csv", index=False)
        print(f"[Fold {fold}] HGB={rows[-4]['PR_AUC_AP']:.5f} XGB={rows[-3]['PR_AUC_AP']:.5f} "
              f"fixed50={rows[-2]['PR_AUC_AP']:.5f} nested={rows[-1]['PR_AUC_AP']:.5f}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out / "fold_results.csv", index=False)
    summary = result.groupby("Experiment").agg(PR_AUC_mean=("PR_AUC_AP", "mean"), PR_AUC_std=("PR_AUC_AP", "std"),
                                                AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"),
                                                XGBWeight_mean=("XGBWeight", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(out / "oof_all.csv", index=False)
    pooled = pd.DataFrame([
        {"Experiment": name,
         "Pooled_OOF_PR_AUC": average_precision_score(oof.y_true, oof[f"p_{name}"]),
         "Pooled_OOF_AUROC": roc_auc_score(oof.y_true, oof[f"p_{name}"]),
         "Pooled_OOF_Brier": brier_score_loss(oof.y_true, oof[f"p_{name}"])}
        for name in result.Experiment.drop_duplicates()
    ])
    pooled.to_csv(out / "pooled_oof_summary.csv", index=False)
    comparisons = [("ensemble_nested", "hgb_tuned"), ("ensemble_nested", "ensemble_fixed_50"),
                   ("ensemble_nested", "xgb_meps_tuned")]
    bootstrap = patient_bootstrap(oof, comparisons, args.bootstrap, args.seed)
    bootstrap.to_csv(out / "patient_bootstrap.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPooled OOF\n", pooled.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default=MEPS_BRIDGE)
    parser.add_argument("--outdir", default=result_dir("meps_nested_hgb_blend"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--blend-weights", nargs="+", type=float, default=list(BLEND_WEIGHTS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if (args.folds < 2 or args.bootstrap < 1 or args.xgb_estimators < 1
            or any(w < 0 or w > 1 for w in args.blend_weights)):
        parser.error("Invalid folds, bootstrap, or blend weight")
    run(args)


if __name__ == "__main__":
    main()
