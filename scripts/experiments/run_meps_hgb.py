#!/usr/bin/env python3
"""Evaluate discharge-time HGB with and without external MEPS proxies."""

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from _paths import MEPS_BRIDGE, UCI_RAW, result_dir
from run_cdc_encoder_control import external_meps, load_meps, prepare_hospital


EXPERIMENTS = {
    "Clinical_HGB": [],
    "MEPS_HGB": ["meps_er_8", "meps_ip_8"],
}


def raw_discharge_predictors(frame):
    """Return the 46 encounter-complete UCI predictors used at discharge."""
    excluded = {"encounter_id", "patient_nbr", "readmitted", "target", "weight"}
    columns = [
        name for name in frame.columns
        if name not in excluded
        and not name.startswith("bridge_")
        and not name.startswith("clinical_")
        and not name.startswith("meps_")
    ]
    if len(columns) != 46:
        raise ValueError(f"Expected 46 raw discharge predictors, found {len(columns)}")
    return columns


def normalize(frame, columns):
    output = frame.loc[:, columns].copy()
    categorical = []
    categorical_ids = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    for name in columns:
        if (
            name in categorical_ids
            or pd.api.types.is_object_dtype(output[name])
            or pd.api.types.is_string_dtype(output[name])
        ):
            categorical.append(name)
            output[name] = output[name].astype(object)
            output[name] = output[name].where(pd.notna(output[name]), np.nan)
        else:
            output[name] = pd.to_numeric(output[name], errors="coerce")
    numeric = [name for name in columns if name not in categorical]
    return output, categorical, numeric


def make_preprocessor(categorical, numeric, min_frequency):
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_frequency,
                sparse_output=False,
                dtype=np.float32,
            )),
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric),
    ], sparse_threshold=0., verbose_feature_names_out=False)


def hgb(seed, max_iter, early_stopping):
    return HistGradientBoostingClassifier(
        learning_rate=.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.,
        early_stopping=early_stopping,
        validation_fraction=.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def patient_bootstrap(oof, repetitions, seed):
    patients, inverse = np.unique(oof.patient_nbr.to_numpy(), return_inverse=True)
    labels = oof.y_true.to_numpy()
    clinical = oof.p_Clinical_HGB.to_numpy()
    external = oof.p_MEPS_HGB.to_numpy()
    observed = average_precision_score(labels, external) - average_precision_score(labels, clinical)
    rng = np.random.default_rng(seed + 9000)
    deltas = []
    for _ in range(repetitions):
        counts = np.bincount(
            rng.integers(0, len(patients), size=len(patients)), minlength=len(patients),
        )
        weight = counts[inverse]
        deltas.append(
            average_precision_score(labels, external, sample_weight=weight)
            - average_precision_score(labels, clinical, sample_weight=weight)
        )
    return pd.DataFrame([{
        "Comparison": "MEPS_HGB - Clinical_HGB",
        "PooledOOF_DeltaAP": observed,
        "CI95_Lower": float(np.quantile(deltas, .025)),
        "CI95_Upper": float(np.quantile(deltas, .975)),
        "FractionPositive": float(np.mean(np.asarray(deltas) > 0)),
        "Bootstraps": repetitions,
    }])


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    frame, _ = prepare_hospital(args.hospital)
    frame = external_meps(
        load_meps(args.meps), frame, args.seed, output, device=args.device,
    )
    base_columns = raw_discharge_predictors(frame)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed,
    )
    results, oof_parts = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(
        frame, frame.target, groups=frame.patient_nbr,
    ), 1):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient leakage detected")
        fold_oof = test[["encounter_id", "patient_nbr", "target"]].rename(
            columns={"target": "y_true"},
        ).copy()
        fold_oof["Fold"] = fold
        for experiment, extra in EXPERIMENTS.items():
            columns = base_columns + extra
            x_train, categorical, numeric = normalize(train, columns)
            x_test, _, _ = normalize(test, columns)
            preprocessor = make_preprocessor(categorical, numeric, args.min_frequency)
            train_matrix = preprocessor.fit_transform(x_train).astype(np.float32, copy=False)
            test_matrix = preprocessor.transform(x_test).astype(np.float32, copy=False)
            model = hgb(args.seed + fold, args.max_iter, args.early_stopping)
            model.fit(train_matrix, train.target.to_numpy())
            probability = model.predict_proba(test_matrix)[:, 1]
            row = {
                "Fold": fold, "Experiment": experiment,
                "PR_AUC_AP": float(average_precision_score(test.target, probability)),
                "AUROC": float(roc_auc_score(test.target, probability)),
                "Brier": float(brier_score_loss(test.target, probability)),
                "Iterations": int(model.n_iter_),
                "EncodedFeatures": int(train_matrix.shape[1]),
            }
            results.append(row)
            fold_oof[f"p_{experiment}"] = probability
            print(
                f"[Fold {fold}] {experiment:12s} PR_AUC={row['PR_AUC_AP']:.5f} "
                f"AUROC={row['AUROC']:.5f} iterations={row['Iterations']} "
                f"features={row['EncodedFeatures']}", flush=True,
            )
            del train_matrix, test_matrix
        oof_parts.append(fold_oof)
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)
    table = pd.DataFrame(results)
    table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        PR_AUC_mean=("PR_AUC_AP", "mean"), PR_AUC_std=("PR_AUC_AP", "std"),
        AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"),
        Iterations_mean=("Iterations", "mean"),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output / "oof_all.csv", index=False)
    bootstrap = patient_bootstrap(oof, args.bootstrap, args.seed)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default=MEPS_BRIDGE)
    parser.add_argument("--outdir", default=result_dir("meps_hgb_results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 1 or args.max_iter < 1:
        parser.error("folds >= 2, bootstrap >= 1, and max_iter >= 1 are required")
    run(args)


if __name__ == "__main__":
    main()
