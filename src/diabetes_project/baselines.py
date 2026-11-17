"""One reusable runner for LR, random forest, and XGBoost baselines."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .metrics import best_f1_threshold, binary_metrics
from .plots import save_diagnostic_curves


def _model(name: str, seed: int, scale_pos_weight: float, options: dict):
    if name == "lr":
        return LogisticRegression(
            max_iter=options.get("max_iter", 2000), class_weight="balanced",
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=options.get("n_estimators", 300),
            max_depth=options.get("max_depth", 16), min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=options.get("n_jobs", -1),
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("Install the xgboost optional dependency to run xgb") from error
        return XGBClassifier(
            n_estimators=options.get("n_estimators", 500), max_depth=options.get("max_depth", 4),
            learning_rate=options.get("learning_rate", 0.05), subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", random_state=seed, n_jobs=options.get("n_jobs", -1),
            scale_pos_weight=scale_pos_weight,
        )
    raise ValueError(f"Unknown model: {name}")


def run_baselines(fold_dir: str | Path, output_dir: str | Path, models, seed=42, **options):
    source, output = Path(fold_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrices = {name: sparse.load_npz(source / f"{name}.npz")
                for name in ("train", "inner_train", "val", "test")}
    rows = {name: pd.read_csv(source / f"{name}_rows.csv") for name in matrices}
    labels = {name: frame["target"].to_numpy(dtype=int) for name, frame in rows.items()}
    positive = labels["inner_train"].sum()
    ratio = (len(labels["inner_train"]) - positive) / max(positive, 1)
    results = []
    for name in models:
        # Inner features have their own training-fitted vocabulary. Use them only
        # to select the threshold, then refit on the outer-train representation
        # before evaluating the untouched outer test fold.
        tuning_estimator = _model(name, seed, ratio, options)
        tuning_estimator.fit(matrices["inner_train"], labels["inner_train"])
        val_probability = tuning_estimator.predict_proba(matrices["val"])[:, 1]
        threshold = best_f1_threshold(labels["val"], val_probability)
        outer_positive = labels["train"].sum()
        outer_ratio = (len(labels["train"]) - outer_positive) / max(outer_positive, 1)
        estimator = _model(name, seed, outer_ratio, options)
        estimator.fit(matrices["train"], labels["train"])
        test_probability = estimator.predict_proba(matrices["test"])[:, 1]
        for split_name, probability in (("validation", val_probability), ("test", test_probability)):
            key = "val" if split_name == "validation" else "test"
            for policy, value in (("default_0.5", 0.5), ("val_f1_optimal", threshold)):
                results.append({"model": name, "split": split_name, "threshold_policy": policy,
                                **binary_metrics(labels[key], probability, value)})
        joblib.dump(estimator, output / f"{name}_model.joblib", compress=3)
        prediction = rows["test"].copy()
        prediction["predicted_probability"] = test_probability
        prediction.to_csv(output / f"{name}_test_predictions.csv", index=False)
        save_diagnostic_curves(labels["test"], test_probability, output, name)
    table = pd.DataFrame(results)
    table.to_csv(output / "metrics.csv", index=False)
    (output / "run_summary.json").write_text(json.dumps({
        "fold_dir": str(source.resolve()), "models": list(models), "seed": seed,
        "fit_partition": "inner_train", "threshold_partition": "val", "final_partition": "test",
    }, indent=2), encoding="utf-8")
    return table
