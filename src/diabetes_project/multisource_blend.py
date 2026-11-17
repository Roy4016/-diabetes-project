"""Multi-source MEPS auxiliary learning with nested Clinical XGBoost blending."""

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
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .data import load_uci_diabetes
from .event_transfer import make_event_features
from .external_embedding import _append_embedding, _patient_bootstrap
from .longitudinal_blend import _fit_xgb, _metrics, _tune_xgb
from .longitudinal_meps import (
    LongitudinalConfig,
    LongitudinalNet,
    _external_losses,
    _paired_patient_bootstrap,
    _reduce,
    _seed,
    _tensor,
    load_longitudinal_meps,
    make_uci_temporal_features,
)
from .metrics import best_f1_threshold


@dataclass
class MultisourceBlendConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    event_csv: str
    output_dir: str
    max_folds: int = 5
    seed: int = 777
    svd_components: int = 64
    embedding_dim: int = 16
    pretrain_epochs: int = 40
    uci_epochs: int = 12
    patience: int = 4
    batch_size: int = 4096
    meps_batch_size: int = 512
    event_batch_size: int = 256
    learning_rate: float = 1e-3
    lambda_meps: float = .2
    lambda_event: float = .1
    event_pretrain_weight: float = .5
    expense_weight: float = .2
    er_weight: float = .2
    xgb_estimators: int = 300
    xgb_early_stopping: int = 25
    max_external_weight: float = .6
    blend_steps: int = 13
    min_blend_gain: float = 0.
    bootstrap: int = 1000
    device: str = "cpu"


class MultisourceNet(LongitudinalNet):
    def __init__(self, clinical_dim, temporal_dim, event_dim, embedding_dim):
        super().__init__(clinical_dim, temporal_dim, embedding_dim)
        self.event_adapter = nn.Sequential(
            nn.Linear(event_dim, 32), nn.LayerNorm(32), nn.ReLU(),
        )
        self.event_binary = nn.Linear(embedding_dim, 2)

    def event_embedding(self, values):
        return self.shared(self.event_adapter(values))


def _event_loss(labels, device):
    positive = np.nansum(labels, axis=0)
    observed = np.isfinite(labels).sum(axis=0)
    negative = observed - positive
    return nn.BCEWithLogitsLoss(
        pos_weight=_tensor(np.sqrt(negative / np.maximum(positive, 1)), device),
        reduction="none",
    )


def _masked_event_loss(raw, labels, mask, task_weight):
    return (raw * mask * task_weight).sum() / (mask * task_weight).sum().clamp_min(1.)


def _source_split(long_groups, event_groups, seed):
    """Use one person-level source split across both MEPS task families."""
    people = np.unique(np.concatenate([long_groups.astype(str), event_groups.astype(str)]))
    train, validation = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=seed,
    ).split(people, groups=people))
    train_people = set(people[train])
    long_train = np.array([str(value) in train_people for value in long_groups])
    event_train = np.array([str(value) in train_people for value in event_groups])
    return np.flatnonzero(long_train), np.flatnonzero(~long_train), np.flatnonzero(event_train), np.flatnonzero(~event_train)


def _external_state(net):
    return {
        name: copy.deepcopy(getattr(net, name).state_dict())
        for name in (
            "meps_adapter", "event_adapter", "shared", "meps_binary",
            "meps_expense", "event_binary",
        )
    }


def _load_external_state(net, state):
    for name, values in state.items():
        getattr(net, name).load_state_dict(values)
    # Annual MEPS and UCI have the same ordered 14 temporal columns.
    net.uci_adapter.load_state_dict(state["meps_adapter"])


def pretrain_multisource(long_x, long_binary, long_expense, long_groups,
                         event_x, event_y, event_groups, config, shuffled_event=False):
    lt, lv, et, ev = _source_split(long_groups, event_groups, config.seed)
    train_event_y = event_y.copy()
    if shuffled_event:
        order = np.random.default_rng(config.seed + 55).permutation(len(et))
        train_event_y[et] = train_event_y[et][order]
    expense_mean = float(long_expense[lt].mean())
    expense_std = float(long_expense[lt].std() + 1e-6)
    normalized_expense = (long_expense - expense_mean) / expense_std
    _seed(config.seed + (300 if shuffled_event else 100))
    net = MultisourceNet(1, long_x.shape[1], event_x.shape[1], config.embedding_dim).to(config.device)
    names = ("meps_adapter", "event_adapter", "shared", "meps_binary", "meps_expense", "event_binary")
    parameters = [parameter for name in names for parameter in getattr(net, name).parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=1e-4)
    long_binary_loss, expense_loss = _external_losses(long_binary[lt], config.device)
    event_loss = _event_loss(train_event_y[et], config.device)
    event_task_weight = _tensor(np.array([1., config.er_weight], dtype=np.float32), config.device)
    lx, lb = _tensor(long_x[lt], config.device), _tensor(long_binary[lt], config.device)
    le = _tensor(normalized_expense[lt], config.device)
    ex = _tensor(event_x[et], config.device)
    ey = _tensor(np.nan_to_num(train_event_y[et], nan=0.), config.device)
    em = _tensor(np.isfinite(train_event_y[et]).astype(np.float32), config.device)
    vlx, vlb = _tensor(long_x[lv], config.device), _tensor(long_binary[lv], config.device)
    vle = _tensor(normalized_expense[lv], config.device)
    vex = _tensor(event_x[ev], config.device)
    vey = _tensor(np.nan_to_num(event_y[ev], nan=0.), config.device)
    vem = _tensor(np.isfinite(event_y[ev]).astype(np.float32), config.device)
    rng = np.random.default_rng(config.seed + 700)
    best, best_epoch, state, wait = float("inf"), 1, None, 0
    for epoch in range(1, config.pretrain_epochs + 1):
        net.train()
        for batch in torch.randperm(len(lx), device=config.device).split(config.meps_batch_size):
            picked = rng.choice(len(ex), size=min(config.event_batch_size, len(ex)), replace=False)
            eb = torch.as_tensor(picked, device=config.device)
            optimizer.zero_grad()
            long_z = net.meps_embedding(lx[batch])
            loss = long_binary_loss(net.meps_binary(long_z), lb[batch])
            loss += config.expense_weight * expense_loss(net.meps_expense(long_z).squeeze(1), le[batch])
            event_z = net.event_embedding(ex[eb])
            raw = event_loss(net.event_binary(event_z), ey[eb])
            loss += config.event_pretrain_weight * _masked_event_loss(raw, ey[eb], em[eb], event_task_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            long_z = net.meps_embedding(vlx)
            value = long_binary_loss(net.meps_binary(long_z), vlb)
            value += config.expense_weight * expense_loss(net.meps_expense(long_z).squeeze(1), vle)
            event_z = net.event_embedding(vex)
            raw = event_loss(net.event_binary(event_z), vey)
            value += config.event_pretrain_weight * _masked_event_loss(raw, vey, vem, event_task_weight)
            value = float(value.item())
        if value < best - 1e-5:
            best, best_epoch, state, wait = value, epoch, _external_state(net), 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    _load_external_state(net, state)
    net.eval()
    with torch.no_grad():
        lp = torch.sigmoid(net.meps_binary(net.meps_embedding(vlx))).cpu().numpy()
        ep = torch.sigmoid(net.event_binary(net.event_embedding(vex))).cpu().numpy()
    info = {
        "longitudinal_pairs": int(len(long_x)), "event_rows": int(len(event_x)),
        "best_epoch": best_epoch,
        "longitudinal_ER_AP": float(average_precision_score(long_binary[lv, 0], lp[:, 0])),
        "longitudinal_IP_AP": float(average_precision_score(long_binary[lv, 1], lp[:, 1])),
    }
    for column, name in enumerate(("readmission", "ER_revisit")):
        valid = np.isfinite(event_y[ev, column])
        info[f"event_{name}_AP"] = float(average_precision_score(event_y[ev, column][valid], ep[valid, column]))
        info[f"event_{name}_AUROC"] = float(roc_auc_score(event_y[ev, column][valid], ep[valid, column]))
    return state, info, (expense_mean, expense_std), lt, et, train_event_y


def _train_uci(net_state, clinical, temporal, labels, long_x, long_binary, long_expense,
               event_x, event_y, expense_stats, config, seed, epochs, validation=None):
    _seed(seed)
    net = MultisourceNet(clinical.shape[1], temporal.shape[1], event_x.shape[1], config.embedding_dim).to(config.device)
    _load_external_state(net, net_state)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    primary_loss = nn.BCEWithLogitsLoss()
    long_binary_loss, expense_loss = _external_losses(long_binary, config.device)
    event_loss = _event_loss(event_y, config.device)
    event_task_weight = _tensor(np.array([1., config.er_weight], dtype=np.float32), config.device)
    normalized_expense = (long_expense - expense_stats[0]) / expense_stats[1]
    lx, lb = _tensor(long_x, config.device), _tensor(long_binary, config.device)
    le = _tensor(normalized_expense, config.device)
    ex = _tensor(event_x, config.device)
    ey = _tensor(np.nan_to_num(event_y, nan=0.), config.device)
    em = _tensor(np.isfinite(event_y).astype(np.float32), config.device)
    rng = np.random.default_rng(seed + 500)
    best_ap, best_epoch, best_state, wait = -1., 1, None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        batches = np.array_split(
            np.random.default_rng(seed + epoch).permutation(len(labels)),
            max(1, int(np.ceil(len(labels) / config.batch_size))),
        )
        for batch in batches:
            long_idx = torch.as_tensor(rng.choice(len(lx), min(config.meps_batch_size, len(lx)), replace=False), device=config.device)
            event_idx = torch.as_tensor(rng.choice(len(ex), min(config.event_batch_size, len(ex)), replace=False), device=config.device)
            optimizer.zero_grad()
            logits = net.uci_logits(_tensor(clinical[batch], config.device), _tensor(temporal[batch], config.device), True)
            loss = primary_loss(logits, _tensor(labels[batch], config.device))
            long_z = net.meps_embedding(lx[long_idx])
            auxiliary = long_binary_loss(net.meps_binary(long_z), lb[long_idx])
            auxiliary += config.expense_weight * expense_loss(net.meps_expense(long_z).squeeze(1), le[long_idx])
            event_z = net.event_embedding(ex[event_idx])
            event_raw = event_loss(net.event_binary(event_z), ey[event_idx])
            event_auxiliary = _masked_event_loss(event_raw, ey[event_idx], em[event_idx], event_task_weight)
            loss += config.lambda_meps * auxiliary + config.lambda_event * event_auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
        if validation is not None:
            probability = _predict_uci(net, validation[0], validation[1], config.device)
            ap = average_precision_score(validation[2], probability)
            if ap > best_ap + 1e-5:
                best_ap, best_epoch, best_state, wait = ap, epoch, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= config.patience:
                    break
    if validation is not None:
        net.load_state_dict(best_state)
    return net, best_epoch, best_ap


@torch.no_grad()
def _predict_uci(net, clinical, temporal, device, batch_size=4096):
    net.eval()
    output = []
    for start in range(0, len(clinical), batch_size):
        logits = net.uci_logits(
            _tensor(clinical[start:start + batch_size], device),
            _tensor(temporal[start:start + batch_size], device), True,
        )
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def _risk_scores(net, temporal, device, batch_size=4096):
    """Annual risks plus exact-event heads evaluated on the aligned UCI embedding."""
    net.eval()
    output = []
    for start in range(0, len(temporal), batch_size):
        z = net.uci_embedding(_tensor(temporal[start:start + batch_size], device))
        annual = torch.cat([torch.sigmoid(net.meps_binary(z)), net.meps_expense(z)], dim=1)
        event = torch.sigmoid(net.event_binary(z))
        output.append(torch.cat([annual, event], dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def run_multisource_blend(config: MultisourceBlendConfig):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    uci = load_uci_diabetes(config.uci_csv)
    temporal_all = make_uci_temporal_features(uci)
    long_x, long_binary, long_expense, long_groups, _ = load_longitudinal_meps(config.meps_csv)
    event = pd.read_csv(config.event_csv, dtype={"person_key": str})
    event_x = make_event_features(event)
    event_y = event[["readmission_30d", "er_revisit_30d"]].to_numpy(np.float32)
    event_groups = event.person_key.to_numpy()
    print(f"[MEPS] joint pretraining: {len(long_x):,} year pairs + {len(event_x):,} events", flush=True)
    true_state, true_info, true_stats, long_train, event_train, true_event_y = pretrain_multisource(
        long_x, long_binary, long_expense, long_groups, event_x, event_y, event_groups,
        config, shuffled_event=False,
    )
    shuffled_state, shuffled_info, shuffled_stats, shuffled_long_train, shuffled_event_train, shuffled_event_y = pretrain_multisource(
        long_x, long_binary, long_expense, long_groups, event_x, event_y, event_groups,
        config, shuffled_event=True,
    )
    external = {
        "event": (true_state, true_stats, long_train, event_train, true_event_y),
        "shuffled": (
            shuffled_state, shuffled_stats, shuffled_long_train,
            shuffled_event_train, shuffled_event_y,
        ),
    }
    variants = ("clinical", "risk_event", "blend_event", "risk_shuffled", "blend_shuffled")
    fold_dirs = sorted(
        (path for path in Path(config.folds_dir).glob("fold_[0-9]*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    weights = np.linspace(0., config.max_external_weight, config.blend_steps)
    results, oof_parts = [], {name: [] for name in variants}
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        loaded = np.load(fold_dir / "indices.npz")
        idx = {key: loaded[key] for key in loaded.files}
        matrices = {name: sparse.load_npz(fold_dir / f"{name}.npz") for name in ("train", "test", "inner_train", "val")}
        rows = {name: pd.read_csv(fold_dir / f"{name}_rows.csv") for name in matrices}
        labels = {name: rows[name].target.to_numpy(np.float32) for name in rows}
        inner_svd, val_svd = _reduce(matrices["inner_train"], matrices["val"], components=config.svd_components, seed=config.seed + fold)
        full_svd, test_svd = _reduce(matrices["train"], matrices["test"], components=config.svd_components, seed=config.seed + fold)
        temporal = {
            "inner": temporal_all[idx["inner_train_idx"]], "val": temporal_all[idx["val_idx"]],
            "full": temporal_all[idx["train_idx"]], "test": temporal_all[idx["test_idx"]],
        }
        clinical_val, clinical_trees = _tune_xgb(
            matrices["inner_train"], matrices["val"], labels["inner_train"], labels["val"], config.seed + fold, config,
        )
        clinical_threshold = best_f1_threshold(labels["val"], clinical_val)
        clinical_model, clinical_test = _fit_xgb(matrices["train"], matrices["test"], labels["train"], config.seed + fold, clinical_trees, config)
        clinical_validation_ap = average_precision_score(labels["val"], clinical_val)
        results.append({"Fold": fold, "Experiment": "clinical", "Trees": clinical_trees, "ExternalWeight": 0., **_metrics(labels["test"], clinical_test, clinical_threshold)})
        joblib.dump(clinical_model, output / f"clinical_fold{fold}_xgb.joblib", compress=3)
        fold_probability = {"clinical": clinical_test}
        for tag, (state, stats, lt, et, task_event_y) in external.items():
            tuning_net, epochs, encoder_val_ap = _train_uci(
                state, inner_svd, temporal["inner"], labels["inner_train"],
                long_x[lt], long_binary[lt], long_expense[lt], event_x[et], task_event_y[et],
                stats, config, config.seed + fold, config.uci_epochs,
                validation=(val_svd, temporal["val"], labels["val"]),
            )
            inner_risk = _risk_scores(tuning_net, temporal["inner"], config.device)
            val_risk = _risk_scores(tuning_net, temporal["val"], config.device)
            risk_val, risk_trees = _tune_xgb(
                _append_embedding(matrices["inner_train"], inner_risk),
                _append_embedding(matrices["val"], val_risk),
                labels["inner_train"], labels["val"], config.seed + fold, config,
            )
            blend_scores = [average_precision_score(labels["val"], (1-a)*clinical_val + a*risk_val) for a in weights]
            selected = int(np.argmax(blend_scores))
            weight = float(weights[selected])
            if blend_scores[selected] < clinical_validation_ap + config.min_blend_gain:
                weight = 0.
            risk_threshold = best_f1_threshold(labels["val"], risk_val)
            blend_threshold = best_f1_threshold(labels["val"], (1-weight)*clinical_val + weight*risk_val)
            final_net, _, _ = _train_uci(
                state, full_svd, temporal["full"], labels["train"],
                long_x[lt], long_binary[lt], long_expense[lt], event_x[et], task_event_y[et],
                stats, config, config.seed + fold, epochs,
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
            common = {"Fold": fold, "Trees": risk_trees, "EncoderEpochs": epochs, "EncoderValidationAP": encoder_val_ap}
            results.append({**common, "Experiment": risk_name, "ExternalWeight": 1., **_metrics(labels["test"], risk_test, risk_threshold)})
            results.append({**common, "Experiment": blend_name, "ExternalWeight": weight, "BlendValidationAP": blend_scores[selected], "ClinicalValidationAP": clinical_validation_ap, **_metrics(labels["test"], blend_test, blend_threshold)})
            fold_probability[risk_name] = risk_test
            fold_probability[blend_name] = blend_test
            torch.save(final_net.state_dict(), output / f"{tag}_fold{fold}_encoder.pt")
            joblib.dump(risk_model, output / f"{risk_name}_fold{fold}_xgb.joblib", compress=3)
            print(f"[Fold {fold}] {tag:8s} risk_AP={results[-2]['AP']:.5f} blend_AP={results[-1]['AP']:.5f} weight={weight:.2f}", flush=True)
        for name, probability in fold_probability.items():
            part = rows["test"].rename(columns={"target": "y_true"}).copy()
            part["Fold"] = fold
            part["probability"] = probability
            oof_parts[name].append(part)
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)
    table = pd.DataFrame(results)
    table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        F1_mean=("F1", "mean"), Brier_mean=("Brier", "mean"),
        ExternalWeight_mean=("ExternalWeight", "mean"),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    combined = None
    for name, parts in oof_parts.items():
        current = pd.concat(parts, ignore_index=True)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(columns={"probability": f"p_{name}"})
        combined = probability if combined is None else combined.merge(probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one")
    combined.to_csv(output / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    external_signal = _paired_patient_bootstrap(combined, [
        ("risk_event", "risk_shuffled"), ("blend_event", "blend_shuffled"),
    ], config.seed, config.bootstrap)
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "true_external_validation": true_info,
        "shuffled_event_validation": shuffled_info,
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
