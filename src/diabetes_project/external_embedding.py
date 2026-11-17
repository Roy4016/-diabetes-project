"""MEPS-supervised embeddings combined with a clinical XGBoost classifier."""

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
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from .data import load_uci_diabetes
from .features import diagnosis_flag
from .metrics import best_f1_threshold

RACES = ("Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other")
VARIANTS = ("clinical", "random", "pretrained", "multitask", "shuffled")


@dataclass
class EmbeddingExperimentConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    output_dir: str
    variants: tuple[str, ...] = VARIANTS
    max_folds: int = 5
    seed: int = 777
    embedding_dim: int = 16
    pretrain_epochs: int = 30
    multitask_epochs: int = 20
    patience: int = 5
    batch_size: int = 1024
    meps_batch_size: int = 512
    learning_rate: float = 1e-3
    lambda_meps: float = 0.2
    xgb_estimators: int = 300
    xgb_early_stopping: int = 25
    bootstrap: int = 200
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _age_midpoint(series: pd.Series) -> pd.Series:
    bounds = series.astype("string").str.extract(r"^\[(\d+)-(\d+)\)$")
    return (pd.to_numeric(bounds[0], errors="coerce") + pd.to_numeric(bounds[1], errors="coerce")) / 2


def _race_columns(series: pd.Series) -> list[np.ndarray]:
    cleaned = series.astype("string").str.strip()
    return [(cleaned == race).fillna(False).to_numpy(dtype=np.float32) for race in RACES]


def make_uci_shared_features(frame: pd.DataFrame) -> np.ndarray:
    """Create harmonized UCI inputs without using the readmission label."""
    age = (_age_midpoint(frame["age"]) / 100).fillna(0.5).to_numpy(dtype=np.float32)
    male = frame["gender"].astype("string").str.lower().map({"male": 1.0, "female": 0.0})
    disease = [
        diagnosis_flag(frame, 401, 405),
        diagnosis_flag(frame, 272, 272),
        diagnosis_flag(frame, 430, 438),
        diagnosis_flag(frame, 410, 414),
        diagnosis_flag(frame, 410, 410) | diagnosis_flag(frame, 412, 412),
    ]
    columns = [
        age,
        male.fillna(-1).to_numpy(dtype=np.float32),
        *[values.astype(np.float32) for values in disease],
        *_race_columns(frame["race"]),
    ]
    return np.column_stack(columns).astype(np.float32)


def load_meps_shared_features(path: str | Path):
    frame = pd.read_csv(path, low_memory=False, dtype={"person_key": str})
    required = {
        "person_key", "age", "sex", "high_bp", "high_chol", "stroke",
        "heart_disease", "mi", "er_visits", "inpatient_nights",
        "er_any", "inpatient_any", "race_common",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MEPS dataset is missing columns: {sorted(missing)}")
    numeric = lambda name: pd.to_numeric(frame[name], errors="coerce")
    columns = [
        (numeric("age") / 100).fillna(0.5).clip(0, 1).to_numpy(dtype=np.float32),
        numeric("sex").map({1: 1.0, 2: 0.0}).fillna(-1).to_numpy(dtype=np.float32),
        *[numeric(name).where(numeric(name).isin([0, 1]), -1).to_numpy(dtype=np.float32)
          for name in ("high_bp", "high_chol", "stroke", "heart_disease", "mi")],
        *_race_columns(frame["race_common"]),
    ]
    features = np.column_stack(columns).astype(np.float32)
    labels = frame[["er_any", "inpatient_any"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(labels).all(axis=1) & np.isin(labels, [0, 1]).all(axis=1)
    return features[valid], labels[valid], frame.loc[valid, "person_key"].to_numpy()


class SharedExternalNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, embedding_dim), nn.ReLU(),
        )
        self.meps_head = nn.Linear(embedding_dim, 2)
        self.uci_head = nn.Linear(embedding_dim, 1)

    def embedding(self, values):
        return self.encoder(values)


def _tensor(values, device):
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _balanced_loss(labels: np.ndarray, device):
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    weight = np.sqrt(negative / np.maximum(positive, 1))
    return nn.BCEWithLogitsLoss(pos_weight=_tensor(weight, device))


@torch.no_grad()
def encode(net: SharedExternalNet, values: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    net.eval()
    output = []
    for start in range(0, len(values), batch_size):
        output.append(net.embedding(_tensor(values[start:start + batch_size], device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def pretrain_meps(features, labels, groups, config: EmbeddingExperimentConfig):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
    train_idx, val_idx = next(splitter.split(features, groups=groups))
    if set(groups[train_idx]).intersection(groups[val_idx]):
        raise ValueError("MEPS person leakage detected")
    set_seed(config.seed + 100)
    net = SharedExternalNet(features.shape[1], config.embedding_dim).to(config.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = _balanced_loss(labels[train_idx], config.device)
    best_loss, best_epoch, best_state, wait = float("inf"), 1, None, 0
    x_train, y_train = _tensor(features[train_idx], config.device), _tensor(labels[train_idx], config.device)
    x_val, y_val = _tensor(features[val_idx], config.device), _tensor(labels[val_idx], config.device)
    for epoch in range(1, config.pretrain_epochs + 1):
        net.train()
        for batch in torch.randperm(len(x_train), device=config.device).split(config.meps_batch_size):
            optimizer.zero_grad()
            loss = loss_fn(net.meps_head(net.embedding(x_train[batch])), y_train[batch])
            loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net.meps_head(net.embedding(x_val)), y_val).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        probability = torch.sigmoid(net.meps_head(net.embedding(x_val))).cpu().numpy()
    diagnostics = {
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "validation_people": int(len(np.unique(groups[val_idx]))),
        "ER_AP": float(average_precision_score(labels[val_idx, 0], probability[:, 0])),
        "ER_AUROC": float(roc_auc_score(labels[val_idx, 0], probability[:, 0])),
        "IP_AP": float(average_precision_score(labels[val_idx, 1], probability[:, 1])),
        "IP_AUROC": float(roc_auc_score(labels[val_idx, 1], probability[:, 1])),
    }
    return net, diagnostics


def train_multitask_encoder(
    initial: SharedExternalNet,
    uci_train_x, uci_train_y, uci_val_x, uci_val_y,
    meps_x, meps_y, config: EmbeddingExperimentConfig, seed: int,
    shuffle_external: bool = False,
):
    set_seed(seed)
    net = copy.deepcopy(initial).to(config.device)
    if shuffle_external:
        meps_y = meps_y[np.random.default_rng(seed).permutation(len(meps_y))]
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    uci_loss = _balanced_loss(uci_train_y.reshape(-1, 1), config.device)
    meps_loss = _balanced_loss(meps_y, config.device)
    ux, uy = _tensor(uci_train_x, config.device), _tensor(uci_train_y, config.device)
    mx, my = _tensor(meps_x, config.device), _tensor(meps_y, config.device)
    best_ap, best_epoch, best_state, wait = -1.0, 1, None, 0
    rng = np.random.default_rng(seed + 500)
    for epoch in range(1, config.multitask_epochs + 1):
        net.train()
        for batch in torch.randperm(len(ux), device=config.device).split(config.batch_size):
            picked = rng.choice(len(mx), size=min(config.meps_batch_size, len(mx)), replace=False)
            meps_batch = torch.as_tensor(picked, device=config.device)
            optimizer.zero_grad()
            readmission_logits = net.uci_head(net.embedding(ux[batch])).squeeze(1)
            external_logits = net.meps_head(net.embedding(mx[meps_batch]))
            loss = uci_loss(readmission_logits, uy[batch]) + config.lambda_meps * meps_loss(
                external_logits, my[meps_batch]
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); optimizer.step()
        val_probability = torch.sigmoid(
            net.uci_head(net.embedding(_tensor(uci_val_x, config.device))).squeeze(1)
        ).detach().cpu().numpy()
        ap = average_precision_score(uci_val_y, val_probability)
        if ap > best_ap + 1e-5:
            best_ap, best_epoch, best_state, wait = ap, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    net.load_state_dict(best_state)
    return net, {"best_epoch": best_epoch, "validation_encoder_AP": float(best_ap)}


def _xgb(seed, estimators, early_stopping=None, device="cpu"):
    return XGBClassifier(
        n_estimators=estimators, max_depth=4, learning_rate=0.04,
        min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=3.0, objective="binary:logistic", eval_metric="aucpr",
        tree_method="hist", device=device, random_state=seed, n_jobs=-1,
        early_stopping_rounds=early_stopping,
    )


def _append_embedding(matrix, embedding):
    return sparse.hstack([matrix, sparse.csr_matrix(embedding)], format="csr")


def _evaluate_variant(name, fold_dir, indices, embedding, config, fold):
    matrices = {part: sparse.load_npz(fold_dir / f"{part}.npz")
                for part in ("train", "test", "inner_train", "val")}
    rows = {part: pd.read_csv(fold_dir / f"{part}_rows.csv")
            for part in matrices}
    y = {part: rows[part]["target"].to_numpy(dtype=int) for part in rows}
    if embedding is not None:
        matrix_indices = {
            "train": indices["train_idx"], "test": indices["test_idx"],
            "inner_train": indices["inner_train_idx"], "val": indices["val_idx"],
        }
        matrices = {part: _append_embedding(matrix, embedding[matrix_indices[part]])
                    for part, matrix in matrices.items()}
    tuning = _xgb(config.seed + fold, config.xgb_estimators, config.xgb_early_stopping, config.device)
    tuning.fit(matrices["inner_train"], y["inner_train"], eval_set=[(matrices["val"], y["val"])], verbose=False)
    validation_probability = tuning.predict_proba(matrices["val"])[:, 1]
    threshold = best_f1_threshold(y["val"], validation_probability)
    best_trees = int(getattr(tuning, "best_iteration", config.xgb_estimators - 1)) + 1
    final = _xgb(config.seed + fold, best_trees, None, config.device)
    final.fit(matrices["train"], y["train"], verbose=False)
    probability = final.predict_proba(matrices["test"])[:, 1]
    prediction = probability >= threshold
    metrics = {
        "Fold": fold, "Experiment": name, "AP": float(average_precision_score(y["test"], probability)),
        "AUROC": float(roc_auc_score(y["test"], probability)),
        "F1": float(f1_score(y["test"], prediction, zero_division=0)),
        "Brier": float(brier_score_loss(y["test"], probability)),
        "Threshold": threshold, "Trees": best_trees,
    }
    output = rows["test"].rename(columns={"target": "y_true"}).copy()
    output["Fold"] = fold; output["probability"] = probability
    return metrics, output, final


def _patient_bootstrap(oof: pd.DataFrame, variants, seed: int, count: int) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame()
    patients, inverse = np.unique(oof["patient_nbr"].to_numpy(), return_inverse=True)
    y = oof["y_true"].to_numpy()
    rng = np.random.default_rng(seed + 999)
    baseline = oof["p_clinical"].to_numpy()
    result = []
    for variant in variants:
        if variant == "clinical":
            continue
        probability = oof[f"p_{variant}"].to_numpy()
        observed = average_precision_score(y, probability) - average_precision_score(y, baseline)
        deltas = []
        for _ in range(count):
            drawn = rng.integers(0, len(patients), size=len(patients))
            weights = np.bincount(drawn, minlength=len(patients))[inverse]
            deltas.append(
                average_precision_score(y, probability, sample_weight=weights)
                - average_precision_score(y, baseline, sample_weight=weights)
            )
        result.append({
            "Comparison": f"{variant} - clinical", "PooledOOF_DeltaAP": observed,
            "CI95_Lower": float(np.quantile(deltas, 0.025)),
            "CI95_Upper": float(np.quantile(deltas, 0.975)),
            "FractionPositive": float(np.mean(np.asarray(deltas) > 0)), "Bootstraps": count,
        })
    return pd.DataFrame(result)


def run_embedding_experiment(config: EmbeddingExperimentConfig):
    output_dir = Path(config.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if not set(config.variants).issubset(VARIANTS):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    uci = load_uci_diabetes(config.uci_csv)
    uci_shared = make_uci_shared_features(uci)
    meps_x, meps_y, meps_groups = load_meps_shared_features(config.meps_csv)
    pretrained, external_diagnostics = pretrain_meps(meps_x, meps_y, meps_groups, config)
    torch.save(pretrained.state_dict(), output_dir / "meps_pretrained_encoder.pt")
    random_net = SharedExternalNet(uci_shared.shape[1], config.embedding_dim).to(config.device)
    set_seed(config.seed + 200); random_net.apply(
        lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None
    )
    fixed_embeddings = {
        "random": encode(random_net, uci_shared, config.device),
        "pretrained": encode(pretrained, uci_shared, config.device),
    }
    results, predictions = [], {variant: [] for variant in config.variants}
    fold_dirs = [path for path in Path(config.folds_dir).glob("fold_[0-9]*") if path.is_dir()]
    fold_dirs.sort(key=lambda path: int(path.name.split("_")[-1]))
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1])
        loaded = np.load(fold_dir / "indices.npz")
        indices = {key: loaded[key] for key in loaded.files}
        train_ids, val_ids = indices["inner_train_idx"], indices["val_idx"]
        fold_embeddings = dict(fixed_embeddings)
        encoder_info = {}
        for variant, shuffled in (("multitask", False), ("shuffled", True)):
            if variant not in config.variants:
                continue
            net, info = train_multitask_encoder(
                pretrained, uci_shared[train_ids], uci.iloc[train_ids]["target"].to_numpy(),
                uci_shared[val_ids], uci.iloc[val_ids]["target"].to_numpy(),
                meps_x, meps_y, config, config.seed + fold, shuffled,
            )
            fold_embeddings[variant] = encode(net, uci_shared, config.device)
            torch.save(net.state_dict(), output_dir / f"{variant}_fold{fold}_encoder.pt")
            encoder_info[variant] = info
        for variant in config.variants:
            metrics, prediction, model = _evaluate_variant(
                variant, fold_dir, indices, fold_embeddings.get(variant), config, fold
            )
            results.append(metrics); predictions[variant].append(prediction)
            joblib.dump(model, output_dir / f"{variant}_fold{fold}_xgb.joblib", compress=3)
            print(f"[Fold {fold}] {variant:10s} AP={metrics['AP']:.5f} AUROC={metrics['AUROC']:.5f} F1={metrics['F1']:.5f}", flush=True)
        (output_dir / f"encoder_fold{fold}.json").write_text(json.dumps(encoder_info, indent=2), encoding="utf-8")
        pd.DataFrame(results).to_csv(output_dir / "partial_results.csv", index=False)
    table = pd.DataFrame(results); table.to_csv(output_dir / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"), F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Brier_mean=("Brier", "mean"), Brier_std=("Brier", "std"),
    ).reset_index(); summary.to_csv(output_dir / "summary.csv", index=False)
    combined = None
    for variant, frames in predictions.items():
        current = pd.concat(frames, ignore_index=True)
        current.to_csv(output_dir / f"oof_{variant}.csv", index=False)
        probability = current[["encounter_id", "patient_nbr", "y_true", "Fold", "probability"]].rename(
            columns={"probability": f"p_{variant}"}
        )
        combined = probability if combined is None else combined.merge(
            probability, on=["encounter_id", "patient_nbr", "y_true", "Fold"], validate="one_to_one"
        )
    combined.to_csv(output_dir / "oof_all.csv", index=False)
    bootstrap = _patient_bootstrap(combined, config.variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output_dir / "patient_bootstrap.csv", index=False)
    metadata = {"config": asdict(config), "external_validation": external_diagnostics,
                "uci_rows": len(uci), "uci_patients": int(uci.patient_nbr.nunique()),
                "meps_rows": len(meps_x), "meps_people": int(len(np.unique(meps_groups))),
                "shared_feature_count": int(uci_shared.shape[1])}
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary, bootstrap
