"""Transfer an exact MEPS 30-day event risk score into Clinical XGBoost."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .data import load_uci_diabetes
from .external_embedding import RACES, _evaluate_variant, _patient_bootstrap
from .features import diagnosis_flag
from .longitudinal_meps import _paired_patient_bootstrap, _tensor


VARIANTS = ("clinical", "random_risk", "event_pretrained", "event_shuffled")


@dataclass
class EventTransferConfig:
    folds_dir: str
    uci_csv: str
    event_csv: str
    output_dir: str
    variants: tuple[str, ...] = VARIANTS
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
    bootstrap: int = 1000
    device: str = "cpu"


def _seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def _race(series):
    values = series.astype("string")
    return [(values == race).fillna(False).to_numpy(dtype=np.float32) for race in RACES]


def make_event_features(frame):
    numeric = lambda name: pd.to_numeric(frame[name], errors="coerce")
    disease = [numeric(name).where(numeric(name).isin([0, 1]), -1).to_numpy(np.float32)
               for name in ("high_bp", "high_chol", "stroke", "heart_disease", "mi")]
    columns = [
        (numeric("age").fillna(50).clip(0, 100) / 100).to_numpy(np.float32),
        numeric("sex").map({1: 1., 2: 0.}).fillna(-1).to_numpy(np.float32),
        *disease, *_race(frame["race_common"]),
        (np.log1p(numeric("length_of_stay").fillna(0).clip(0, 30)) / np.log(31)).to_numpy(np.float32),
        numeric("emergency_admission").fillna(-1).to_numpy(np.float32),
        numeric("any_operation").fillna(-1).to_numpy(np.float32),
    ]
    return np.column_stack(columns).astype(np.float32)


def make_uci_event_features(frame):
    age = frame["age"].astype("string").str.extract(r"^\[(\d+)-(\d+)\)$")
    age = ((pd.to_numeric(age[0], errors="coerce") + pd.to_numeric(age[1], errors="coerce")) / 200).fillna(.5)
    male = frame["gender"].astype("string").str.lower().map({"male": 1., "female": 0.}).fillna(-1)
    disease = [
        diagnosis_flag(frame, 401, 405), diagnosis_flag(frame, 272, 272),
        diagnosis_flag(frame, 430, 438), diagnosis_flag(frame, 410, 414),
        diagnosis_flag(frame, 410, 410) | diagnosis_flag(frame, 412, 412),
    ]
    source = pd.to_numeric(frame["admission_source_id"], errors="coerce")
    procedures = pd.to_numeric(frame["num_procedures"], errors="coerce")
    stay = pd.to_numeric(frame["time_in_hospital"], errors="coerce")
    columns = [
        age.to_numpy(np.float32), male.to_numpy(np.float32),
        *[values.astype(np.float32) for values in disease], *_race(frame["race"]),
        (np.log1p(stay.fillna(0).clip(0, 30)) / np.log(31)).to_numpy(np.float32),
        source.eq(7).astype(np.float32).to_numpy(),
        procedures.gt(0).astype(np.float32).to_numpy(),
    ]
    return np.column_stack(columns).astype(np.float32)


class EventRiskNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, embedding_dim), nn.ReLU(),
        )
        self.head = nn.Linear(embedding_dim, 2)

    def forward(self, values):
        return self.head(self.encoder(values))


def _loss(labels, device):
    positive = np.nansum(labels, axis=0); observed = np.isfinite(labels).sum(axis=0)
    negative = observed - positive
    return nn.BCEWithLogitsLoss(
        pos_weight=_tensor(np.sqrt(negative / np.maximum(positive, 1)), device), reduction="none",
    )


def pretrain_event(features, labels, groups, config, shuffled=False):
    train_idx, val_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=config.seed,
    ).split(features, groups=groups))
    train_labels = labels.copy()
    if shuffled:
        order = np.random.default_rng(config.seed + 55).permutation(len(train_idx))
        train_labels[train_idx] = train_labels[train_idx][order]
    _seed(config.seed + (300 if shuffled else 100))
    net = EventRiskNet(features.shape[1], config.embedding_dim).to(config.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = _loss(train_labels[train_idx], config.device)
    x_train = _tensor(features[train_idx], config.device)
    y_train = _tensor(np.nan_to_num(train_labels[train_idx], nan=0.), config.device)
    mask_train = _tensor(np.isfinite(train_labels[train_idx]).astype(np.float32), config.device)
    x_val = _tensor(features[val_idx], config.device)
    y_val = _tensor(np.nan_to_num(labels[val_idx], nan=0.), config.device)
    mask_val = _tensor(np.isfinite(labels[val_idx]).astype(np.float32), config.device)
    task_weight = _tensor(np.array([1., config.er_weight], dtype=np.float32), config.device)
    best, state, epoch_best, wait = float("inf"), None, 1, 0
    for epoch in range(1, config.pretrain_epochs + 1):
        net.train()
        for batch in torch.randperm(len(x_train), device=config.device).split(config.batch_size):
            optimizer.zero_grad(); raw = criterion(net(x_train[batch]), y_train[batch])
            loss = (raw * mask_train[batch] * task_weight).sum() / (mask_train[batch] * task_weight).sum()
            loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad():
            raw = criterion(net(x_val), y_val)
            value = ((raw * mask_val * task_weight).sum() / (mask_val * task_weight).sum()).item()
        if value < best - 1e-5:
            best, state, epoch_best, wait = value, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        probability = torch.sigmoid(net(x_val)).cpu().numpy()
    info = {"events": len(features), "train_events": len(train_idx), "validation_events": len(val_idx),
            "best_epoch": epoch_best}
    for column, name in enumerate(("readmission", "er_revisit")):
        valid = np.isfinite(labels[val_idx, column])
        info[f"{name}_AP"] = float(average_precision_score(labels[val_idx, column][valid], probability[valid, column]))
        info[f"{name}_AUROC"] = float(roc_auc_score(labels[val_idx, column][valid], probability[valid, column]))
    return net, info


@torch.no_grad()
def risk_score(net, values, device, batch_size=4096):
    net.eval(); output = []
    for start in range(0, len(values), batch_size):
        logits = net(_tensor(values[start:start + batch_size], device))[:, 0]
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).reshape(-1, 1).astype(np.float32)


def run_event_transfer(config: EventTransferConfig):
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    event = pd.read_csv(config.event_csv, dtype={"person_key": str})
    event_x = make_event_features(event)
    event_y = event[["readmission_30d", "er_revisit_30d"]].to_numpy(np.float32)
    true_net, true_info = pretrain_event(event_x, event_y, event.person_key.to_numpy(), config, False)
    shuffled_net, shuffled_info = pretrain_event(event_x, event_y, event.person_key.to_numpy(), config, True)
    uci = load_uci_diabetes(config.uci_csv); uci_x = make_uci_event_features(uci)
    _seed(config.seed + 200); random_net = EventRiskNet(uci_x.shape[1], config.embedding_dim).to(config.device)
    embeddings = {
        "random_risk": risk_score(random_net, uci_x, config.device),
        "event_pretrained": risk_score(true_net, uci_x, config.device),
        "event_shuffled": risk_score(shuffled_net, uci_x, config.device),
    }
    torch.save(true_net.state_dict(), output / "event_pretrained.pt")
    torch.save(shuffled_net.state_dict(), output / "event_shuffled.pt")
    fold_dirs = sorted(Path(config.folds_dir).glob("fold_[0-9]*"), key=lambda p: int(p.name.split("_")[-1]))
    results, predictions = [], {name: [] for name in config.variants}
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1]); loaded = np.load(fold_dir / "indices.npz")
        indices = {key: loaded[key] for key in loaded.files}
        for variant in config.variants:
            metrics, prediction, model = _evaluate_variant(
                variant, fold_dir, indices, embeddings.get(variant), config, fold,
            )
            results.append(metrics); predictions[variant].append(prediction)
            joblib.dump(model, output / f"{variant}_fold{fold}_xgb.joblib", compress=3)
            print(f"[Fold {fold}] {variant:17s} AP={metrics['AP']:.5f} AUROC={metrics['AUROC']:.5f}", flush=True)
    table = pd.DataFrame(results); table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        F1_mean=("F1", "mean"), Brier_mean=("Brier", "mean"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)
    combined = None
    for variant, parts in predictions.items():
        current = pd.concat(parts, ignore_index=True)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(
            columns={"probability": f"p_{variant}"})
        combined = probability if combined is None else combined.merge(
            probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one")
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, config.variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    external_signal = _paired_patient_bootstrap(
        combined, [("event_pretrained", "event_shuffled")], config.seed, config.bootstrap)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "true_external_validation": true_info,
        "shuffled_external_validation": shuffled_info,
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
