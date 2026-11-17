"""Nested blending of Clinical XGBoost and longitudinal MEPS risk models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from .data import load_uci_diabetes
from .external_embedding import _append_embedding, _patient_bootstrap, _xgb
from .longitudinal_meps import (
    LongitudinalConfig, _paired_patient_bootstrap, _reduce, _train_uci,
    load_longitudinal_meps, make_uci_temporal_features, pretrain_longitudinal,
)
from .longitudinal_xgb import _risk_scores
from .metrics import best_f1_threshold


@dataclass
class LongitudinalBlendConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    output_dir: str
    max_folds: int = 5
    seed: int = 777
    svd_components: int = 64
    embedding_dim: int = 16
    pretrain_epochs: int = 30
    uci_epochs: int = 12
    patience: int = 3
    batch_size: int = 4096
    meps_batch_size: int = 512
    learning_rate: float = 1e-3
    lambda_meps: float = .2
    lambda_values: tuple[float, ...] = ()
    expense_weight: float = .2
    xgb_estimators: int = 300
    xgb_early_stopping: int = 25
    max_external_weight: float = .6
    blend_steps: int = 13
    min_blend_gain: float = 0.
    bootstrap: int = 1000
    device: str = "cpu"


def _metrics(labels, probability, threshold):
    return {
        "AP": float(average_precision_score(labels, probability)),
        "AUROC": float(roc_auc_score(labels, probability)),
        "F1": float(f1_score(labels, probability >= threshold, zero_division=0)),
        "Brier": float(brier_score_loss(labels, probability)),
    }


def _tune_xgb(train, validation, y_train, y_validation, seed, config):
    model = _xgb(seed, config.xgb_estimators, config.xgb_early_stopping, config.device)
    model.fit(train, y_train, eval_set=[(validation, y_validation)], verbose=False)
    probability = model.predict_proba(validation)[:, 1]
    trees = int(getattr(model, "best_iteration", config.xgb_estimators - 1)) + 1
    return probability, trees


def _fit_xgb(train, test, labels, seed, trees, config):
    model = _xgb(seed, trees, None, config.device)
    model.fit(train, labels, verbose=False)
    return model, model.predict_proba(test)[:, 1]


def run_longitudinal_blend(config: LongitudinalBlendConfig):
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    uci = load_uci_diabetes(config.uci_csv)
    temporal_all = make_uci_temporal_features(uci)
    meps_x, meps_binary, meps_expense, groups, _ = load_longitudinal_meps(config.meps_csv)
    neural_config = LongitudinalConfig(
        folds_dir=config.folds_dir, uci_csv=config.uci_csv, meps_csv=config.meps_csv,
        output_dir=config.output_dir, seed=config.seed, svd_components=config.svd_components,
        embedding_dim=config.embedding_dim, pretrain_epochs=config.pretrain_epochs,
        uci_epochs=config.uci_epochs, patience=config.patience, batch_size=config.batch_size,
        meps_batch_size=config.meps_batch_size, learning_rate=config.learning_rate,
        lambda_meps=config.lambda_meps, expense_weight=config.expense_weight,
        bootstrap=config.bootstrap, device=config.device,
    )
    print(f"[MEPS] pretraining true and shuffled longitudinal encoders ({len(meps_x):,} pairs)", flush=True)
    true_state, true_info, true_stats, true_train = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, neural_config, shuffled=False,
    )
    shuffled_state, shuffled_info, shuffled_stats, shuffled_train = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, neural_config, shuffled=True,
    )
    shuffled_binary = meps_binary.copy(); shuffled_expense = meps_expense.copy()
    order = np.random.default_rng(config.seed + 55).permutation(len(shuffled_train))
    shuffled_binary[shuffled_train] = shuffled_binary[shuffled_train][order]
    shuffled_expense[shuffled_train] = shuffled_expense[shuffled_train][order]
    external = {
        "mtl": (true_state, true_stats, true_train, meps_binary, meps_expense, "aligned_mtl"),
        "shuffled_mtl": (
            shuffled_state, shuffled_stats, shuffled_train,
            shuffled_binary, shuffled_expense, "aligned_shuffled_mtl",
        ),
    }

    variants = ("clinical", "risk_mtl", "blend_mtl", "risk_shuffled_mtl", "blend_shuffled_mtl")
    fold_dirs = sorted(
        (p for p in Path(config.folds_dir).glob("fold_[0-9]*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    results, oof_parts = [], {name: [] for name in variants}
    weights = np.linspace(0, config.max_external_weight, config.blend_steps)
    lambda_values = config.lambda_values or (config.lambda_meps,)
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        loaded = np.load(fold_dir / "indices.npz")
        idx = {key: loaded[key] for key in loaded.files}
        matrices = {name: sparse.load_npz(fold_dir / f"{name}.npz")
                    for name in ("train", "test", "inner_train", "val")}
        rows = {name: pd.read_csv(fold_dir / f"{name}_rows.csv") for name in matrices}
        labels = {name: rows[name].target.to_numpy(dtype=np.float32) for name in rows}
        inner_svd, val_svd = _reduce(
            matrices["inner_train"], matrices["val"],
            components=config.svd_components, seed=config.seed + fold,
        )
        full_svd, test_svd = _reduce(
            matrices["train"], matrices["test"],
            components=config.svd_components, seed=config.seed + fold,
        )
        temporal = {
            "inner": temporal_all[idx["inner_train_idx"]], "val": temporal_all[idx["val_idx"]],
            "full": temporal_all[idx["train_idx"]], "test": temporal_all[idx["test_idx"]],
        }
        clinical_val, clinical_trees = _tune_xgb(
            matrices["inner_train"], matrices["val"], labels["inner_train"], labels["val"],
            config.seed + fold, config,
        )
        clinical_threshold = best_f1_threshold(labels["val"], clinical_val)
        clinical_model, clinical_test = _fit_xgb(
            matrices["train"], matrices["test"], labels["train"],
            config.seed + fold, clinical_trees, config,
        )
        clinical_row = {"Fold": fold, "Experiment": "clinical", "Trees": clinical_trees,
                        "ExternalWeight": 0., **_metrics(labels["test"], clinical_test, clinical_threshold)}
        results.append(clinical_row)
        joblib.dump(clinical_model, output / f"clinical_fold{fold}_xgb.joblib", compress=3)
        fold_probabilities = {"clinical": clinical_test}

        clinical_validation_ap = average_precision_score(labels["val"], clinical_val)
        for tag, (state, stats, train_external, ext_binary, ext_expense, neural_variant) in external.items():
            candidates = []
            for lambda_value in lambda_values:
                candidate_config = replace(neural_config, lambda_meps=lambda_value)
                tuning_net, epochs, encoder_val_ap = _train_uci(
                    neural_variant, inner_svd, temporal["inner"], labels["inner_train"],
                    meps_x[train_external], ext_binary[train_external], ext_expense[train_external],
                    state, stats, candidate_config, config.seed + fold, config.uci_epochs,
                    validation=(val_svd, temporal["val"], labels["val"]),
                )
                inner_risk = _risk_scores(tuning_net, temporal["inner"], config.device)
                val_risk = _risk_scores(tuning_net, temporal["val"], config.device)
                risk_val, risk_trees = _tune_xgb(
                    _append_embedding(matrices["inner_train"], inner_risk),
                    _append_embedding(matrices["val"], val_risk),
                    labels["inner_train"], labels["val"], config.seed + fold, config,
                )
                scores = [
                    average_precision_score(labels["val"], (1-a)*clinical_val + a*risk_val)
                    for a in weights
                ]
                candidates.append({
                    "lambda": lambda_value, "config": candidate_config, "net": tuning_net,
                    "epochs": epochs, "encoder_val_ap": encoder_val_ap,
                    "risk_val": risk_val, "risk_trees": risk_trees, "scores": scores,
                    "raw_weight": float(weights[int(np.argmax(scores))]),
                    "validation_ap": float(max(scores)),
                })
            selected = max(candidates, key=lambda item: item["validation_ap"])
            epochs, encoder_val_ap = selected["epochs"], selected["encoder_val_ap"]
            risk_val, risk_trees, scores = selected["risk_val"], selected["risk_trees"], selected["scores"]
            weight = selected["raw_weight"]
            if selected["validation_ap"] < clinical_validation_ap + config.min_blend_gain:
                weight = 0.
            blend_val = (1-weight)*clinical_val + weight*risk_val
            blend_threshold = best_f1_threshold(labels["val"], blend_val)
            risk_threshold = best_f1_threshold(labels["val"], risk_val)

            final_net, _, _ = _train_uci(
                neural_variant, full_svd, temporal["full"], labels["train"],
                meps_x[train_external], ext_binary[train_external], ext_expense[train_external],
                state, stats, selected["config"], config.seed + fold, epochs,
            )
            train_risk = _risk_scores(final_net, temporal["full"], config.device)
            test_risk = _risk_scores(final_net, temporal["test"], config.device)
            risk_model, risk_test = _fit_xgb(
                _append_embedding(matrices["train"], train_risk),
                _append_embedding(matrices["test"], test_risk), labels["train"],
                config.seed + fold, risk_trees, config,
            )
            blend_test = (1-weight)*clinical_test + weight*risk_test
            risk_name, blend_name = f"risk_{tag}", f"blend_{tag}"
            common = {"Fold": fold, "Trees": risk_trees, "EncoderEpochs": epochs,
                      "EncoderValidationAP": encoder_val_ap,
                      "LambdaMEPS": selected["lambda"]}
            results.append({**common, "Experiment": risk_name, "ExternalWeight": 1.,
                            **_metrics(labels["test"], risk_test, risk_threshold)})
            results.append({**common, "Experiment": blend_name, "ExternalWeight": weight,
                            "BlendValidationAP": selected["validation_ap"],
                            "ClinicalValidationAP": clinical_validation_ap,
                            **_metrics(labels["test"], blend_test, blend_threshold)})
            fold_probabilities[risk_name] = risk_test; fold_probabilities[blend_name] = blend_test
            torch.save(final_net.state_dict(), output / f"{tag}_fold{fold}_encoder.pt")
            joblib.dump(risk_model, output / f"{risk_name}_fold{fold}_xgb.joblib", compress=3)
            print(
                f"[Fold {fold}] {tag:12s} risk_AP={results[-2]['AP']:.5f} "
                f"blend_AP={results[-1]['AP']:.5f} weight={weight:.2f} "
                f"lambda={selected['lambda']:.3g}", flush=True,
            )

        for name, probability in fold_probabilities.items():
            part = rows["test"].rename(columns={"target": "y_true"}).copy()
            part["Fold"] = fold; part["probability"] = probability
            oof_parts[name].append(part)
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)

    table = pd.DataFrame(results); table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        F1_mean=("F1", "mean"), Brier_mean=("Brier", "mean"),
        ExternalWeight_mean=("ExternalWeight", "mean"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)
    combined = None
    for name, parts in oof_parts.items():
        current = pd.concat(parts, ignore_index=True)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(
            columns={"probability": f"p_{name}"}
        )
        combined = probability if combined is None else combined.merge(
            probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one",
        )
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    external_signal = _paired_patient_bootstrap(combined, [
        ("risk_mtl", "risk_shuffled_mtl"), ("blend_mtl", "blend_shuffled_mtl"),
    ], config.seed, config.bootstrap)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "true_external_validation": true_info,
        "shuffled_external_validation": shuffled_info,
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
