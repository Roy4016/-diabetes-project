"""Append leakage-safe longitudinal MEPS representations to Clinical XGBoost."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import load_uci_diabetes
from .external_embedding import _evaluate_variant, _patient_bootstrap
from .longitudinal_meps import (
    LongitudinalConfig, LongitudinalNet, _load_external_state,
    _paired_patient_bootstrap, _tensor, load_longitudinal_meps,
    make_uci_temporal_features, pretrain_longitudinal,
)

VARIANTS = (
    "clinical", "random", "aligned_pretrained", "aligned_shuffled",
    "aligned_mtl", "aligned_shuffled_mtl",
    "risk_pretrained", "risk_shuffled", "risk_mtl", "risk_shuffled_mtl",
)


@dataclass
class LongitudinalXGBConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    encoder_dir: str
    output_dir: str
    variants: tuple[str, ...] = VARIANTS
    max_folds: int = 5
    seed: int = 777
    embedding_dim: int = 16
    pretrain_epochs: int = 30
    patience: int = 3
    meps_batch_size: int = 512
    learning_rate: float = 1e-3
    expense_weight: float = .2
    xgb_estimators: int = 300
    xgb_early_stopping: int = 25
    bootstrap: int = 1000
    device: str = "cpu"


@torch.no_grad()
def _encode(net, values, device, batch_size=4096):
    net.eval(); output = []
    for start in range(0, len(values), batch_size):
        output.append(net.uci_embedding(_tensor(values[start:start + batch_size], device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def _risk_scores(net, values, device, batch_size=4096):
    """Return interpretable MEPS ER, inpatient, and expense risk scores."""
    net.eval(); output = []
    for start in range(0, len(values), batch_size):
        embedding = net.uci_embedding(_tensor(values[start:start + batch_size], device))
        binary = torch.sigmoid(net.meps_binary(embedding))
        expense = net.meps_expense(embedding)
        output.append(torch.cat([binary, expense], dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _net_from_external(state, temporal_dim, config, align=True):
    net = LongitudinalNet(1, temporal_dim, config.embedding_dim).to(config.device)
    _load_external_state(net, state, align_uci=align)
    return net


def _load_fold_encoder(path, temporal_dim, config):
    # The upstream longitudinal experiment uses 64 SVD clinical components by
    # default. Only uci_adapter/shared are consumed here, so the clinical branch
    # does not contribute predictions or expose labels to the outer test fold.
    state = torch.load(path, map_location=config.device, weights_only=True)
    clinical_dim = state["clinical.0.weight"].shape[1]
    embedding_dim = state["shared.0.weight"].shape[0]
    net = LongitudinalNet(clinical_dim, temporal_dim, embedding_dim).to(config.device)
    net.load_state_dict(state)
    return net


def run_longitudinal_xgb(config: LongitudinalXGBConfig):
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    if not set(config.variants).issubset(VARIANTS):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    if "clinical" not in config.variants:
        raise ValueError("clinical must be included as the bootstrap baseline")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    uci = load_uci_diabetes(config.uci_csv)
    temporal = make_uci_temporal_features(uci)
    meps_x, meps_binary, meps_expense, groups, _ = load_longitudinal_meps(config.meps_csv)
    pretrain_config = LongitudinalConfig(
        folds_dir=config.folds_dir, uci_csv=config.uci_csv, meps_csv=config.meps_csv,
        output_dir=config.output_dir, seed=config.seed, embedding_dim=config.embedding_dim,
        pretrain_epochs=config.pretrain_epochs, patience=config.patience,
        meps_batch_size=config.meps_batch_size, learning_rate=config.learning_rate,
        expense_weight=config.expense_weight, device=config.device,
    )
    print(f"[MEPS] pretraining longitudinal encoders from {len(meps_x):,} year pairs", flush=True)
    true_state, true_info, _, _ = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, pretrain_config, shuffled=False,
    )
    shuffled_state, shuffled_info, _, _ = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, pretrain_config, shuffled=True,
    )
    true_net = _net_from_external(true_state, temporal.shape[1], config)
    shuffled_net = _net_from_external(shuffled_state, temporal.shape[1], config)
    torch.manual_seed(config.seed + 200)
    random_net = LongitudinalNet(1, temporal.shape[1], config.embedding_dim).to(config.device)
    fixed = {
        "random": _encode(random_net, temporal, config.device),
        "aligned_pretrained": _encode(true_net, temporal, config.device),
        "aligned_shuffled": _encode(shuffled_net, temporal, config.device),
        "risk_pretrained": _risk_scores(true_net, temporal, config.device),
        "risk_shuffled": _risk_scores(shuffled_net, temporal, config.device),
    }
    torch.save(true_state, output / "meps_longitudinal_pretrained.pt")
    torch.save(shuffled_state, output / "meps_longitudinal_shuffled.pt")

    fold_dirs = sorted(
        (p for p in Path(config.folds_dir).glob("fold_[0-9]*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    encoder_dir = Path(config.encoder_dir)
    results, predictions = [], {name: [] for name in config.variants}
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        loaded = np.load(fold_dir / "indices.npz")
        indices = {key: loaded[key] for key in loaded.files}
        embeddings = dict(fixed)
        sources = {
            "aligned_mtl": encoder_dir / f"aligned_mtl_fold{fold}.pt",
            "aligned_shuffled_mtl": encoder_dir / f"aligned_shuffled_mtl_fold{fold}.pt",
            "risk_mtl": encoder_dir / f"aligned_mtl_fold{fold}.pt",
            "risk_shuffled_mtl": encoder_dir / f"aligned_shuffled_mtl_fold{fold}.pt",
        }
        for name, path in sources.items():
            if name in config.variants:
                if not path.exists():
                    raise FileNotFoundError(f"Run the aligned longitudinal experiment first: {path}")
                net = _load_fold_encoder(path, temporal.shape[1], config)
                embeddings[name] = (
                    _risk_scores(net, temporal, config.device)
                    if name.startswith("risk_") else _encode(net, temporal, config.device)
                )
        for variant in config.variants:
            metrics, prediction, model = _evaluate_variant(
                variant, fold_dir, indices, embeddings.get(variant), config, fold,
            )
            results.append(metrics); predictions[variant].append(prediction)
            joblib.dump(model, output / f"{variant}_fold{fold}_xgb.joblib", compress=3)
            print(
                f"[Fold {fold}] {variant:22s} AP={metrics['AP']:.5f} "
                f"AUROC={metrics['AUROC']:.5f} trees={metrics['Trees']}", flush=True,
            )
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)

    table = pd.DataFrame(results); table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"),
        AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
        F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Brier_mean=("Brier", "mean"), Brier_std=("Brier", "std"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)

    combined = None
    for variant, parts in predictions.items():
        current = pd.concat(parts, ignore_index=True)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(
            columns={"probability": f"p_{variant}"}
        )
        combined = probability if combined is None else combined.merge(
            probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one",
        )
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, config.variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    pairs = [
        pair for pair in [
            ("aligned_pretrained", "aligned_shuffled"),
            ("aligned_mtl", "aligned_shuffled_mtl"),
            ("risk_pretrained", "risk_shuffled"),
            ("risk_mtl", "risk_shuffled_mtl"),
        ] if all(name in config.variants for name in pair)
    ]
    external_signal = _paired_patient_bootstrap(combined, pairs, config.seed, config.bootstrap)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "true_external_validation": true_info,
        "shuffled_external_validation": shuffled_info,
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
