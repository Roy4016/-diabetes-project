"""Nested blending of Clinical XGBoost and exact MEPS event risk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .data import load_uci_diabetes
from .event_transfer import (
    EventRiskNet,
    make_event_features,
    make_uci_event_features,
    pretrain_event,
    risk_score,
)
from .external_embedding import _patient_bootstrap
from .longitudinal_blend import _fit_xgb, _metrics, _tune_xgb
from .longitudinal_meps import _paired_patient_bootstrap
from .metrics import best_f1_threshold


VARIANTS = (
    "clinical",
    "risk_random",
    "blend_random",
    "risk_event",
    "blend_event",
    "risk_shuffled",
    "blend_shuffled",
)


@dataclass
class EventBlendConfig:
    folds_dir: str
    uci_csv: str
    event_csv: str
    output_dir: str
    max_folds: int = 5
    seed: int = 777
    embedding_dim: int = 16
    pretrain_epochs: int = 50
    patience: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    er_weight: float = .2
    xgb_estimators: int = 300
    xgb_early_stopping: int = 25
    max_external_weight: float = .6
    blend_steps: int = 13
    min_blend_gain: float = 0.
    bootstrap: int = 1000
    device: str = "cpu"


def _calibrator(values, labels, seed):
    """Fit a UCI-only Platt calibrator around a fixed external score."""
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(values.reshape(-1, 1), labels)
    return model


def _probability(model, values):
    return model.predict_proba(values.reshape(-1, 1))[:, 1]


def run_event_blend(config: EventBlendConfig):
    if config.blend_steps < 2:
        raise ValueError("blend_steps must be at least 2")
    if not 0 <= config.max_external_weight <= 1:
        raise ValueError("max_external_weight must be between 0 and 1")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event = pd.read_csv(config.event_csv, dtype={"person_key": str})
    event_x = make_event_features(event)
    event_y = event[["readmission_30d", "er_revisit_30d"]].to_numpy(np.float32)
    print(f"[MEPS] pretraining true and shuffled event models ({len(event_x):,} events)", flush=True)
    true_net, true_info = pretrain_event(
        event_x, event_y, event.person_key.to_numpy(), config, shuffled=False,
    )
    shuffled_net, shuffled_info = pretrain_event(
        event_x, event_y, event.person_key.to_numpy(), config, shuffled=True,
    )

    uci = load_uci_diabetes(config.uci_csv)
    uci_x = make_uci_event_features(uci)
    torch.manual_seed(config.seed + 200)
    random_net = EventRiskNet(uci_x.shape[1], config.embedding_dim).to(config.device)
    scores = {
        "random": risk_score(random_net, uci_x, config.device).ravel(),
        "event": risk_score(true_net, uci_x, config.device).ravel(),
        "shuffled": risk_score(shuffled_net, uci_x, config.device).ravel(),
    }
    torch.save(true_net.state_dict(), output / "event_pretrained.pt")
    torch.save(shuffled_net.state_dict(), output / "event_shuffled.pt")

    fold_dirs = sorted(
        (path for path in Path(config.folds_dir).glob("fold_[0-9]*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    weights = np.linspace(0., config.max_external_weight, config.blend_steps)
    results = []
    oof_parts = {name: [] for name in VARIANTS}

    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        loaded = np.load(fold_dir / "indices.npz")
        idx = {key: loaded[key] for key in loaded.files}
        matrices = {
            name: sparse.load_npz(fold_dir / f"{name}.npz")
            for name in ("train", "test", "inner_train", "val")
        }
        rows = {name: pd.read_csv(fold_dir / f"{name}_rows.csv") for name in matrices}
        labels = {name: rows[name].target.to_numpy(dtype=int) for name in rows}

        clinical_val, clinical_trees = _tune_xgb(
            matrices["inner_train"], matrices["val"],
            labels["inner_train"], labels["val"], config.seed + fold, config,
        )
        clinical_threshold = best_f1_threshold(labels["val"], clinical_val)
        clinical_model, clinical_test = _fit_xgb(
            matrices["train"], matrices["test"], labels["train"],
            config.seed + fold, clinical_trees, config,
        )
        clinical_validation_ap = average_precision_score(labels["val"], clinical_val)
        results.append({
            "Fold": fold, "Experiment": "clinical", "Trees": clinical_trees,
            "ExternalWeight": 0., "ClinicalValidationAP": clinical_validation_ap,
            **_metrics(labels["test"], clinical_test, clinical_threshold),
        })
        joblib.dump(clinical_model, output / f"clinical_fold{fold}_xgb.joblib", compress=3)
        fold_probability = {"clinical": clinical_test}

        split_idx = {
            "inner_train": idx["inner_train_idx"], "val": idx["val_idx"],
            "train": idx["train_idx"], "test": idx["test_idx"],
        }
        for tag in ("random", "event", "shuffled"):
            tuning_calibrator = _calibrator(
                scores[tag][split_idx["inner_train"]], labels["inner_train"],
                config.seed + fold,
            )
            risk_val = _probability(tuning_calibrator, scores[tag][split_idx["val"]])
            risk_validation_ap = average_precision_score(labels["val"], risk_val)
            blend_scores = [
                average_precision_score(
                    labels["val"], (1. - weight) * clinical_val + weight * risk_val,
                )
                for weight in weights
            ]
            best_index = int(np.argmax(blend_scores))
            weight = float(weights[best_index])
            best_blend_validation_ap = float(blend_scores[best_index])
            if best_blend_validation_ap < clinical_validation_ap + config.min_blend_gain:
                weight = 0.

            blend_val = (1. - weight) * clinical_val + weight * risk_val
            risk_threshold = best_f1_threshold(labels["val"], risk_val)
            blend_threshold = best_f1_threshold(labels["val"], blend_val)
            final_calibrator = _calibrator(
                scores[tag][split_idx["train"]], labels["train"], config.seed + fold,
            )
            risk_test = _probability(final_calibrator, scores[tag][split_idx["test"]])
            blend_test = (1. - weight) * clinical_test + weight * risk_test
            risk_name, blend_name = f"risk_{tag}", f"blend_{tag}"
            common = {
                "Fold": fold, "Trees": clinical_trees,
                "RiskValidationAP": risk_validation_ap,
                "ClinicalValidationAP": clinical_validation_ap,
                "BlendValidationAP": best_blend_validation_ap,
            }
            results.append({
                **common, "Experiment": risk_name, "ExternalWeight": 1.,
                **_metrics(labels["test"], risk_test, risk_threshold),
            })
            results.append({
                **common, "Experiment": blend_name, "ExternalWeight": weight,
                **_metrics(labels["test"], blend_test, blend_threshold),
            })
            fold_probability[risk_name] = risk_test
            fold_probability[blend_name] = blend_test
            joblib.dump(final_calibrator, output / f"{risk_name}_fold{fold}_calibrator.joblib")
            print(
                f"[Fold {fold}] {tag:8s} risk_AP={results[-2]['AP']:.5f} "
                f"blend_AP={results[-1]['AP']:.5f} weight={weight:.2f}", flush=True,
            )

        for name, probability in fold_probability.items():
            part = rows["test"].rename(columns={"target": "y_true"}).copy()
            part["Fold"] = fold
            part["probability"] = probability
            oof_parts[name].append(part)
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)

    table = pd.DataFrame(results)
    table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"),
        AUROC_mean=("AUROC", "mean"), F1_mean=("F1", "mean"),
        Brier_mean=("Brier", "mean"), ExternalWeight_mean=("ExternalWeight", "mean"),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)

    combined = None
    for name, parts in oof_parts.items():
        current = pd.concat(parts, ignore_index=True)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(
            columns={"probability": f"p_{name}"},
        )
        combined = probability if combined is None else combined.merge(
            probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"],
            validate="one_to_one",
        )
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, VARIANTS, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    external_signal = _paired_patient_bootstrap(combined, [
        ("risk_event", "risk_shuffled"),
        ("blend_event", "blend_shuffled"),
    ], config.seed, config.bootstrap)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "true_external_validation": true_info,
        "shuffled_external_validation": shuffled_info,
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
