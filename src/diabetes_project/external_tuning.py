"""Nested tuning for MEPS multi-task embeddings without outer-test selection."""

from __future__ import annotations

import copy
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
from .external_embedding import (
    EmbeddingExperimentConfig, SharedExternalNet, _append_embedding, _balanced_loss,
    _patient_bootstrap, _tensor, _xgb, encode, load_meps_shared_features,
    make_uci_shared_features, pretrain_meps, set_seed,
)
from .metrics import best_f1_threshold


@dataclass
class TuningConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    output_dir: str
    embedding_dims: tuple[int, ...] = (8, 16)
    lambda_values: tuple[float, ...] = (0.05, 0.1, 0.2)
    max_folds: int = 5
    seed: int = 777
    pretrain_epochs: int = 30
    multitask_epochs: int = 20
    patience: int = 5
    batch_size: int = 1024
    meps_batch_size: int = 512
    learning_rate: float = 1e-3
    xgb_estimators: int = 250
    xgb_early_stopping: int = 20
    bootstrap: int = 200
    device: str = "cpu"


def _embedding_config(config: TuningConfig, dimension: int, weight: float) -> EmbeddingExperimentConfig:
    return EmbeddingExperimentConfig(
        folds_dir=config.folds_dir, uci_csv=config.uci_csv, meps_csv=config.meps_csv,
        output_dir=config.output_dir, variants=("multitask",), max_folds=config.max_folds,
        seed=config.seed, embedding_dim=dimension, pretrain_epochs=config.pretrain_epochs,
        multitask_epochs=config.multitask_epochs, patience=config.patience,
        batch_size=config.batch_size, meps_batch_size=config.meps_batch_size,
        learning_rate=config.learning_rate, lambda_meps=weight,
        xgb_estimators=config.xgb_estimators, xgb_early_stopping=config.xgb_early_stopping,
        bootstrap=config.bootstrap, device=config.device,
    )


def _train_candidate(initial, uci_train_x, uci_train_y, uci_val_x, uci_val_y,
                     meps_x, meps_y, config, seed):
    """Train an inner-fold encoder and retain the epoch selected by UCI AP."""
    from .external_embedding import train_multitask_encoder
    return train_multitask_encoder(
        initial, uci_train_x, uci_train_y, uci_val_x, uci_val_y,
        meps_x, meps_y, config, seed, shuffle_external=False,
    )


def _refit_encoder(initial, uci_x, uci_y, meps_x, meps_y, config, seed, epochs):
    """Refit a selected encoder on all outer-training patients for fixed epochs."""
    set_seed(seed)
    net = copy.deepcopy(initial).to(config.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    uci_loss = _balanced_loss(uci_y.reshape(-1, 1), config.device)
    meps_loss = _balanced_loss(meps_y, config.device)
    ux, uy = _tensor(uci_x, config.device), _tensor(uci_y, config.device)
    mx, my = _tensor(meps_x, config.device), _tensor(meps_y, config.device)
    rng = np.random.default_rng(seed + 500)
    for _ in range(max(1, epochs)):
        net.train()
        for batch in torch.randperm(len(ux), device=config.device).split(config.batch_size):
            picked = rng.choice(len(mx), size=min(config.meps_batch_size, len(mx)), replace=False)
            meps_batch = torch.as_tensor(picked, device=config.device)
            optimizer.zero_grad()
            primary = net.uci_head(net.embedding(ux[batch])).squeeze(1)
            auxiliary = net.meps_head(net.embedding(mx[meps_batch]))
            loss = uci_loss(primary, uy[batch]) + config.lambda_meps * meps_loss(auxiliary, my[meps_batch])
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); optimizer.step()
    return net


def _load_fold(fold_dir):
    matrices = {part: sparse.load_npz(fold_dir / f"{part}.npz")
                for part in ("train", "test", "inner_train", "val")}
    rows = {part: pd.read_csv(fold_dir / f"{part}_rows.csv") for part in matrices}
    labels = {part: rows[part]["target"].to_numpy(dtype=int) for part in rows}
    loaded = np.load(fold_dir / "indices.npz")
    indices = {key: loaded[key] for key in loaded.files}
    return matrices, rows, labels, indices


def _tune_xgb(inner, validation, y_inner, y_validation, seed, config):
    model = _xgb(seed, config.xgb_estimators, config.xgb_early_stopping, config.device)
    model.fit(inner, y_inner, eval_set=[(validation, y_validation)], verbose=False)
    probability = model.predict_proba(validation)[:, 1]
    return {
        "validation_AP": float(average_precision_score(y_validation, probability)),
        "validation_AUROC": float(roc_auc_score(y_validation, probability)),
        "threshold": best_f1_threshold(y_validation, probability),
        "trees": int(getattr(model, "best_iteration", config.xgb_estimators - 1)) + 1,
    }


def _final_xgb(train, test, y_train, y_test, seed, trees, threshold, config):
    model = _xgb(seed, trees, None, config.device)
    model.fit(train, y_train, verbose=False)
    probability = model.predict_proba(test)[:, 1]
    return model, probability, {
        "AP": float(average_precision_score(y_test, probability)),
        "AUROC": float(roc_auc_score(y_test, probability)),
        "F1": float(f1_score(y_test, probability >= threshold, zero_division=0)),
        "Brier": float(brier_score_loss(y_test, probability)),
        "Threshold": float(threshold), "Trees": int(trees),
    }


def run_tuned_experiment(config: TuningConfig):
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    uci = load_uci_diabetes(config.uci_csv)
    uci_shared = make_uci_shared_features(uci)
    meps_x, meps_y, meps_groups = load_meps_shared_features(config.meps_csv)
    pretrained, external_info = {}, {}
    for dimension in config.embedding_dims:
        candidate_config = _embedding_config(config, dimension, config.lambda_values[0])
        net, info = pretrain_meps(meps_x, meps_y, meps_groups, candidate_config)
        pretrained[dimension], external_info[str(dimension)] = net, info
        torch.save(net.state_dict(), output / f"meps_pretrained_dim{dimension}.pt")

    fold_dirs = [path for path in Path(config.folds_dir).glob("fold_[0-9]*") if path.is_dir()]
    fold_dirs.sort(key=lambda path: int(path.name.split("_")[-1]))
    fold_results, tuning_rows, oof_parts = [], [], {"clinical": [], "multitask_tuned": []}
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        matrices, rows, labels, indices = _load_fold(fold_dir)
        # Clinical baseline uses exactly the same nested partitions and XGBoost policy.
        clinical_tuning = _tune_xgb(
            matrices["inner_train"], matrices["val"], labels["inner_train"], labels["val"],
            config.seed + fold, config,
        )
        clinical_model, clinical_probability, clinical_metrics = _final_xgb(
            matrices["train"], matrices["test"], labels["train"], labels["test"],
            config.seed + fold, clinical_tuning["trees"], clinical_tuning["threshold"], config,
        )
        clinical_metrics.update({"Fold": fold, "Experiment": "clinical"})
        fold_results.append(clinical_metrics)
        joblib.dump(clinical_model, output / f"clinical_fold{fold}_xgb.joblib", compress=3)

        best = None
        inner_ids, val_ids = indices["inner_train_idx"], indices["val_idx"]
        for dimension in config.embedding_dims:
            for weight in config.lambda_values:
                candidate_config = _embedding_config(config, dimension, weight)
                net, encoder_info = _train_candidate(
                    pretrained[dimension], uci_shared[inner_ids], labels["inner_train"],
                    uci_shared[val_ids], labels["val"], meps_x, meps_y,
                    candidate_config, config.seed + fold,
                )
                inner_embedding = encode(net, uci_shared[inner_ids], config.device)
                val_embedding = encode(net, uci_shared[val_ids], config.device)
                xgb_info = _tune_xgb(
                    _append_embedding(matrices["inner_train"], inner_embedding),
                    _append_embedding(matrices["val"], val_embedding),
                    labels["inner_train"], labels["val"], config.seed + fold, config,
                )
                row = {"Fold": fold, "embedding_dim": dimension, "lambda_meps": weight,
                       **encoder_info, **xgb_info}
                tuning_rows.append(row)
                if best is None or row["validation_AP"] > best["row"]["validation_AP"]:
                    best = {"row": row, "config": candidate_config}

        chosen = best["row"]; chosen_config = best["config"]
        final_encoder = _refit_encoder(
            pretrained[chosen["embedding_dim"]], uci_shared[indices["train_idx"]], labels["train"],
            meps_x, meps_y, chosen_config, config.seed + fold, chosen["best_epoch"],
        )
        train_embedding = encode(final_encoder, uci_shared[indices["train_idx"]], config.device)
        test_embedding = encode(final_encoder, uci_shared[indices["test_idx"]], config.device)
        model, probability, metrics = _final_xgb(
            _append_embedding(matrices["train"], train_embedding),
            _append_embedding(matrices["test"], test_embedding), labels["train"], labels["test"],
            config.seed + fold, chosen["trees"], chosen["threshold"], config,
        )
        metrics.update({"Fold": fold, "Experiment": "multitask_tuned",
                        "embedding_dim": chosen["embedding_dim"], "lambda_meps": chosen["lambda_meps"],
                        "EncoderEpochs": chosen["best_epoch"], "InnerValidationAP": chosen["validation_AP"]})
        fold_results.append(metrics)
        torch.save(final_encoder.state_dict(), output / f"multitask_tuned_fold{fold}_encoder.pt")
        joblib.dump(model, output / f"multitask_tuned_fold{fold}_xgb.joblib", compress=3)

        for name, probability_values in (("clinical", clinical_probability), ("multitask_tuned", probability)):
            part = rows["test"].rename(columns={"target": "y_true"}).copy()
            part["Fold"] = fold; part["probability"] = probability_values
            oof_parts[name].append(part)
        pd.DataFrame(tuning_rows).to_csv(output / "tuning_results_partial.csv", index=False)
        pd.DataFrame(fold_results).to_csv(output / "fold_results_partial.csv", index=False)
        print(
            f"[Fold {fold}] clinical AP={clinical_metrics['AP']:.5f}; tuned AP={metrics['AP']:.5f}; "
            f"selected dim={chosen['embedding_dim']} lambda={chosen['lambda_meps']} "
            f"inner_AP={chosen['validation_AP']:.5f}", flush=True,
        )

    results = pd.DataFrame(fold_results); results.to_csv(output / "fold_results.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / "tuning_results.csv", index=False)
    summary = results.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"), F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Brier_mean=("Brier", "mean"), Brier_std=("Brier", "std"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)
    combined = None
    for name, parts in oof_parts.items():
        current = pd.concat(parts, ignore_index=True)
        current.to_csv(output / f"oof_{name}.csv", index=False)
        current = current.rename(columns={"probability": f"p_{name}"})
        combined = current if combined is None else combined.merge(
            current, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one"
        )
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(
        combined.rename(columns={"p_multitask_tuned": "p_multitask"}),
        ("clinical", "multitask"), config.seed, config.bootstrap,
    )
    bootstrap["Comparison"] = "multitask_tuned - clinical"
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "external_validation": external_info,
        "selection_rule": "Maximum augmented-XGBoost AP on inner validation only",
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, pd.DataFrame(tuning_rows)
