"""Longitudinal MEPS multi-task learning with domain adapters and gated fusion."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data import load_uci_diabetes
from .external_embedding import RACES, _patient_bootstrap
from .features import diagnosis_flag
from .metrics import best_f1_threshold

VARIANTS = (
    "uci_only", "direct_shared", "pretrained", "longitudinal_mtl", "shuffled_mtl",
    "aligned_pretrained", "aligned_mtl", "aligned_shuffled_mtl",
)


@dataclass
class LongitudinalConfig:
    folds_dir: str
    uci_csv: str
    meps_csv: str
    output_dir: str
    variants: tuple[str, ...] = VARIANTS
    max_folds: int = 5
    seed: int = 777
    svd_components: int = 64
    embedding_dim: int = 16
    pretrain_epochs: int = 40
    uci_epochs: int = 20
    patience: int = 5
    batch_size: int = 1024
    meps_batch_size: int = 512
    learning_rate: float = 1e-3
    lambda_meps: float = 0.2
    expense_weight: float = 0.2
    bootstrap: int = 200
    device: str = "cpu"


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor(values, device):
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _race(series):
    values = series.astype("string").str.strip()
    return [(values == category).fillna(False).to_numpy(dtype=np.float32) for category in RACES]


def _uci_age(series):
    bounds = series.astype("string").str.extract(r"^\[(\d+)-(\d+)\)$")
    return ((pd.to_numeric(bounds[0], errors="coerce") + pd.to_numeric(bounds[1], errors="coerce")) / 200).fillna(.5)


def make_uci_temporal_features(frame):
    disease = [
        diagnosis_flag(frame, 401, 405), diagnosis_flag(frame, 272, 272),
        diagnosis_flag(frame, 430, 438), diagnosis_flag(frame, 410, 414),
        diagnosis_flag(frame, 410, 410) | diagnosis_flag(frame, 412, 412),
    ]
    emergency = np.log1p(pd.to_numeric(frame["number_emergency"], errors="coerce").fillna(0).clip(0, 20)) / np.log(21)
    inpatient = np.log1p(pd.to_numeric(frame["number_inpatient"], errors="coerce").fillna(0).clip(0, 20)) / np.log(21)
    columns = [
        _uci_age(frame["age"]).to_numpy(dtype=np.float32),
        frame["gender"].astype("string").str.lower().map({"male": 1., "female": 0.}).fillna(-1).to_numpy(dtype=np.float32),
        *[x.astype(np.float32) for x in disease], emergency.to_numpy(dtype=np.float32),
        inpatient.to_numpy(dtype=np.float32), *_race(frame["race"]),
    ]
    return np.column_stack(columns).astype(np.float32)


def load_longitudinal_meps(path):
    frame = pd.read_csv(path, low_memory=False, dtype={"person_key": str})
    frame = frame.sort_values(["person_key", "year"]).reset_index(drop=True)
    required = {
        "person_key", "year", "age", "sex", "high_bp", "high_chol", "stroke",
        "heart_disease", "mi", "race_common", "er_visits", "inpatient_nights",
        "er_any", "inpatient_any", "total_expense",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MEPS longitudinal data is missing: {sorted(missing)}")
    grouped = frame.groupby("person_key", sort=False)
    next_year = grouped["year"].shift(-1)
    paired = next_year.eq(frame["year"] + 1)
    numeric = lambda name: pd.to_numeric(frame[name], errors="coerce")
    inputs = np.column_stack([
        (numeric("age") / 100).fillna(.5).clip(0, 1).to_numpy(dtype=np.float32),
        numeric("sex").map({1: 1., 2: 0.}).fillna(-1).to_numpy(dtype=np.float32),
        *[numeric(name).where(numeric(name).isin([0, 1]), -1).to_numpy(dtype=np.float32)
          for name in ("high_bp", "high_chol", "stroke", "heart_disease", "mi")],
        (np.log1p(numeric("er_visits").fillna(0).clip(0, 20)) / np.log(21)).to_numpy(dtype=np.float32),
        (np.log1p(numeric("inpatient_nights").fillna(0).clip(0, 20)) / np.log(21)).to_numpy(dtype=np.float32),
        *_race(frame["race_common"]),
    ]).astype(np.float32)
    binary = np.column_stack([
        grouped["er_any"].shift(-1), grouped["inpatient_any"].shift(-1),
    ]).astype(np.float32)
    expense = np.log1p(pd.to_numeric(grouped["total_expense"].shift(-1), errors="coerce")).to_numpy(dtype=np.float32)
    valid = paired.to_numpy() & np.isfinite(binary).all(axis=1) & np.isfinite(expense)
    valid &= np.isin(binary, [0, 1]).all(axis=1)
    return inputs[valid], binary[valid], expense[valid], frame.loc[valid, "person_key"].to_numpy(), frame.loc[valid, "year"].to_numpy()


class LongitudinalNet(nn.Module):
    def __init__(self, clinical_dim, temporal_dim, embedding_dim):
        super().__init__()
        self.uci_adapter = nn.Sequential(nn.Linear(temporal_dim, 32), nn.LayerNorm(32), nn.ReLU())
        self.meps_adapter = nn.Sequential(nn.Linear(temporal_dim, 32), nn.LayerNorm(32), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(32, embedding_dim), nn.ReLU())
        self.clinical = nn.Sequential(
            nn.Linear(clinical_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(.1),
        )
        self.gate = nn.Sequential(nn.Linear(64 + embedding_dim, embedding_dim), nn.Sigmoid())
        self.readmission = nn.Sequential(
            nn.Linear(64 + embedding_dim, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1),
        )
        self.meps_binary = nn.Linear(embedding_dim, 2)
        self.meps_expense = nn.Linear(embedding_dim, 1)

    def meps_embedding(self, values):
        return self.shared(self.meps_adapter(values))

    def uci_embedding(self, values):
        return self.shared(self.uci_adapter(values))

    def uci_logits(self, clinical, temporal, use_external=True):
        clinical_z = self.clinical(clinical)
        if use_external:
            external_z = self.uci_embedding(temporal)
            external_z = self.gate(torch.cat([clinical_z, external_z], dim=1)) * external_z
        else:
            external_z = torch.zeros(
                (len(clinical_z), self.readmission[0].in_features - clinical_z.shape[1]),
                dtype=clinical_z.dtype, device=clinical_z.device,
            )
        return self.readmission(torch.cat([clinical_z, external_z], dim=1)).squeeze(1)


def _external_state(net):
    return {name: copy.deepcopy(getattr(net, name).state_dict())
            for name in ("meps_adapter", "shared", "meps_binary", "meps_expense")}


def _load_external_state(net, state, align_uci=False):
    for name, values in state.items():
        getattr(net, name).load_state_dict(values)
    if align_uci:
        # The 14 temporal columns have identical order and meaning in UCI and MEPS.
        # Copying the adapter transfers the part learned from MEPS instead of
        # forcing UCI to relearn a new input mapping from scratch.
        net.uci_adapter.load_state_dict(state["meps_adapter"])


def _external_losses(binary_labels, device):
    positive = binary_labels.sum(axis=0); negative = len(binary_labels) - positive
    weight = np.sqrt(negative / np.maximum(positive, 1))
    return nn.BCEWithLogitsLoss(pos_weight=_tensor(weight, device)), nn.HuberLoss()


def pretrain_longitudinal(x, binary, expense, groups, config, shuffled=False):
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=config.seed)
    train_idx, val_idx = next(split.split(x, groups=groups))
    if shuffled:
        order = np.random.default_rng(config.seed + 55).permutation(len(train_idx))
        binary = binary.copy(); expense = expense.copy()
        binary[train_idx] = binary[train_idx][order]; expense[train_idx] = expense[train_idx][order]
    expense_mean, expense_std = float(expense[train_idx].mean()), float(expense[train_idx].std() + 1e-6)
    normalized_expense = (expense - expense_mean) / expense_std
    _seed(config.seed + (300 if shuffled else 100))
    net = LongitudinalNet(1, x.shape[1], config.embedding_dim).to(config.device)
    params = list(net.meps_adapter.parameters()) + list(net.shared.parameters())
    params += list(net.meps_binary.parameters()) + list(net.meps_expense.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    binary_loss, expense_loss = _external_losses(binary[train_idx], config.device)
    tx, tb = _tensor(x[train_idx], config.device), _tensor(binary[train_idx], config.device)
    te = _tensor(normalized_expense[train_idx], config.device)
    vx, vb = _tensor(x[val_idx], config.device), _tensor(binary[val_idx], config.device)
    ve = _tensor(normalized_expense[val_idx], config.device)
    best, state, best_epoch, wait = float("inf"), None, 1, 0
    for epoch in range(1, config.pretrain_epochs + 1):
        net.train()
        for batch in torch.randperm(len(tx), device=config.device).split(config.meps_batch_size):
            optimizer.zero_grad(); z = net.meps_embedding(tx[batch])
            loss = binary_loss(net.meps_binary(z), tb[batch])
            loss += config.expense_weight * expense_loss(net.meps_expense(z).squeeze(1), te[batch])
            loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad():
            z = net.meps_embedding(vx)
            value = binary_loss(net.meps_binary(z), vb)
            value += config.expense_weight * expense_loss(net.meps_expense(z).squeeze(1), ve)
            value = value.item()
        if value < best - 1e-5:
            best, state, best_epoch, wait = value, _external_state(net), epoch, 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    _load_external_state(net, state); net.eval()
    with torch.no_grad():
        z = net.meps_embedding(vx); probability = torch.sigmoid(net.meps_binary(z)).cpu().numpy()
        expense_prediction = net.meps_expense(z).squeeze(1).cpu().numpy() * expense_std + expense_mean
    info = {
        "pairs": int(len(x)), "train_pairs": int(len(train_idx)), "validation_pairs": int(len(val_idx)),
        "best_epoch": best_epoch, "ER_AP": float(average_precision_score(binary[val_idx, 0], probability[:, 0])),
        "ER_AUROC": float(roc_auc_score(binary[val_idx, 0], probability[:, 0])),
        "IP_AP": float(average_precision_score(binary[val_idx, 1], probability[:, 1])),
        "IP_AUROC": float(roc_auc_score(binary[val_idx, 1], probability[:, 1])),
        "expense_log_MAE": float(np.mean(np.abs(expense_prediction - expense[val_idx]))),
        "expense_mean": expense_mean, "expense_std": expense_std,
    }
    return state, info, (expense_mean, expense_std), train_idx


def _reduce(train, *others, components, seed):
    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=seed)
    train_values = svd.fit_transform(train)
    scaler = StandardScaler().fit(train_values)
    return [scaler.transform(train_values).astype(np.float32)] + [
        scaler.transform(svd.transform(value)).astype(np.float32) for value in others
    ]


@torch.no_grad()
def _predict(net, clinical, temporal, use_external, device, batch_size=4096):
    net.eval(); output = []
    for start in range(0, len(clinical), batch_size):
        stop = start + batch_size
        logits = net.uci_logits(_tensor(clinical[start:stop], device), _tensor(temporal[start:stop], device), use_external)
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def _paired_patient_bootstrap(oof, comparisons, seed, count):
    """Compare two transfer variants while resampling whole UCI patients."""
    if count <= 0:
        return pd.DataFrame()
    patients, inverse = np.unique(oof["patient_nbr"].to_numpy(), return_inverse=True)
    labels = oof["y_true"].to_numpy()
    rng = np.random.default_rng(seed + 1999)
    rows = []
    for left, right in comparisons:
        left_probability = oof[f"p_{left}"].to_numpy()
        right_probability = oof[f"p_{right}"].to_numpy()
        observed = average_precision_score(labels, left_probability) - average_precision_score(labels, right_probability)
        deltas = []
        for _ in range(count):
            drawn = rng.integers(0, len(patients), size=len(patients))
            weights = np.bincount(drawn, minlength=len(patients))[inverse]
            deltas.append(
                average_precision_score(labels, left_probability, sample_weight=weights)
                - average_precision_score(labels, right_probability, sample_weight=weights)
            )
        rows.append({
            "Comparison": f"{left} - {right}", "PooledOOF_DeltaAP": observed,
            "CI95_Lower": float(np.quantile(deltas, .025)),
            "CI95_Upper": float(np.quantile(deltas, .975)),
            "FractionPositive": float(np.mean(np.asarray(deltas) > 0)), "Bootstraps": count,
        })
    return pd.DataFrame(rows)


def _train_uci(variant, clinical, temporal, labels, meps_x, meps_binary, meps_expense,
               external_state, expense_stats, config, seed, epochs, validation=None):
    _seed(seed)
    net = LongitudinalNet(clinical.shape[1], temporal.shape[1], config.embedding_dim).to(config.device)
    if external_state is not None:
        _load_external_state(net, external_state, align_uci=variant.startswith("aligned_"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    primary_loss = nn.BCEWithLogitsLoss()
    binary_loss, expense_loss = _external_losses(meps_binary, config.device)
    normalized_expense = (meps_expense - expense_stats[0]) / expense_stats[1]
    mc, mt, my = _tensor(meps_x, config.device), _tensor(meps_binary, config.device), _tensor(normalized_expense, config.device)
    use_external = variant != "uci_only"
    use_auxiliary = variant in {
        "longitudinal_mtl", "shuffled_mtl", "aligned_mtl", "aligned_shuffled_mtl",
    }
    rng = np.random.default_rng(seed + 500)
    best_ap, best_epoch, best_state, wait = -1., 1, None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        for batch in np.array_split(np.random.default_rng(seed + epoch).permutation(len(labels)),
                                    max(1, int(np.ceil(len(labels) / config.batch_size)))):
            optimizer.zero_grad()
            logits = net.uci_logits(_tensor(clinical[batch], config.device), _tensor(temporal[batch], config.device), use_external)
            loss = primary_loss(logits, _tensor(labels[batch], config.device))
            if use_auxiliary:
                picked = rng.choice(len(mc), size=min(config.meps_batch_size, len(mc)), replace=False)
                idx = torch.as_tensor(picked, device=config.device); z = net.meps_embedding(mc[idx])
                auxiliary = binary_loss(net.meps_binary(z), mt[idx])
                auxiliary += config.expense_weight * expense_loss(net.meps_expense(z).squeeze(1), my[idx])
                loss += config.lambda_meps * auxiliary
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
        if validation is not None:
            probability = _predict(net, validation[0], validation[1], use_external, config.device)
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


def run_longitudinal_experiment(config: LongitudinalConfig):
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    uci = load_uci_diabetes(config.uci_csv); uci_temporal = make_uci_temporal_features(uci)
    meps_x, meps_binary, meps_expense, groups, years = load_longitudinal_meps(config.meps_csv)
    print(f"[MEPS] loaded {len(meps_x):,} consecutive-year pairs; pretraining true labels", flush=True)
    true_state, external_info, expense_stats, external_train = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, config, shuffled=False,
    )
    print(f"[MEPS] true-label pretraining complete (best epoch {external_info['best_epoch']})", flush=True)
    print("[MEPS] pretraining shuffled-label negative control", flush=True)
    shuffled_state, shuffled_info, shuffled_stats, shuffled_train = pretrain_longitudinal(
        meps_x, meps_binary, meps_expense, groups, config, shuffled=True,
    )
    print(f"[MEPS] shuffled-label pretraining complete (best epoch {shuffled_info['best_epoch']})", flush=True)
    shuffled_binary = meps_binary.copy(); shuffled_expense = meps_expense.copy()
    shuffled_order = np.random.default_rng(config.seed + 55).permutation(len(shuffled_train))
    shuffled_binary[shuffled_train] = shuffled_binary[shuffled_train][shuffled_order]
    shuffled_expense[shuffled_train] = shuffled_expense[shuffled_train][shuffled_order]
    fold_dirs = [p for p in Path(config.folds_dir).glob("fold_[0-9]*") if p.is_dir()]
    fold_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    results, predictions = [], []
    for fold_dir in fold_dirs[:config.max_folds]:
        fold = int(fold_dir.name.split("_")[-1]); indices_file = np.load(fold_dir / "indices.npz")
        idx = {key: indices_file[key] for key in indices_file.files}
        from scipy import sparse
        matrix = {name: sparse.load_npz(fold_dir / f"{name}.npz") for name in ("train", "test", "inner_train", "val")}
        rows = {name: pd.read_csv(fold_dir / f"{name}_rows.csv") for name in matrix}
        labels = {name: rows[name].target.to_numpy(dtype=np.float32) for name in rows}
        print(f"[Fold {fold}] fitting leakage-safe SVD representations", flush=True)
        inner, val = _reduce(matrix["inner_train"], matrix["val"], components=config.svd_components, seed=config.seed + fold)
        full, test = _reduce(matrix["train"], matrix["test"], components=config.svd_components, seed=config.seed + fold)
        temporal = {
            "inner": uci_temporal[idx["inner_train_idx"]], "val": uci_temporal[idx["val_idx"]],
            "full": uci_temporal[idx["train_idx"]], "test": uci_temporal[idx["test_idx"]],
        }
        fold_output = rows["test"].rename(columns={"target": "y_true"}).copy(); fold_output["Fold"] = fold
        for variant in config.variants:
            shuffled_variant = "shuffled" in variant
            state = None if variant in {"uci_only", "direct_shared"} else (
                shuffled_state if shuffled_variant else true_state
            )
            stats = shuffled_stats if shuffled_variant else expense_stats
            train_external = shuffled_train if shuffled_variant else external_train
            external_binary = shuffled_binary if shuffled_variant else meps_binary
            external_expense = shuffled_expense if shuffled_variant else meps_expense
            _, epochs, validation_ap = _train_uci(
                variant, inner, temporal["inner"], labels["inner_train"],
                meps_x[train_external], external_binary[train_external], external_expense[train_external],
                state, stats, config, config.seed + fold, config.uci_epochs,
                validation=(val, temporal["val"], labels["val"]),
            )
            tuning_model, _, _ = _train_uci(
                variant, inner, temporal["inner"], labels["inner_train"],
                meps_x[train_external], external_binary[train_external], external_expense[train_external],
                state, stats, config, config.seed + fold, epochs,
            )
            val_probability = _predict(tuning_model, val, temporal["val"], variant != "uci_only", config.device)
            threshold = best_f1_threshold(labels["val"], val_probability)
            final, _, _ = _train_uci(
                variant, full, temporal["full"], labels["train"],
                meps_x[train_external], external_binary[train_external], external_expense[train_external],
                state, stats, config, config.seed + fold, epochs,
            )
            probability = _predict(final, test, temporal["test"], variant != "uci_only", config.device)
            row = {
                "Fold": fold, "Experiment": variant,
                "AP": float(average_precision_score(labels["test"], probability)),
                "AUROC": float(roc_auc_score(labels["test"], probability)),
                "F1": float(f1_score(labels["test"], probability >= threshold, zero_division=0)),
                "Brier": float(brier_score_loss(labels["test"], probability)),
                "Threshold": threshold, "Epochs": epochs, "InnerValidationAP": validation_ap,
            }
            results.append(row); fold_output[f"p_{variant}"] = probability
            torch.save(final.state_dict(), output / f"{variant}_fold{fold}.pt")
            print(f"[Fold {fold}] {variant:18s} AP={row['AP']:.5f} AUROC={row['AUROC']:.5f} F1={row['F1']:.5f} epochs={epochs}", flush=True)
        predictions.append(fold_output)
        pd.DataFrame(results).to_csv(output / "partial_results.csv", index=False)
    table = pd.DataFrame(results); table.to_csv(output / "fold_results.csv", index=False)
    summary = table.groupby("Experiment").agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"), AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"), F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Brier_mean=("Brier", "mean"), Brier_std=("Brier", "std"),
    ).reset_index(); summary.to_csv(output / "summary.csv", index=False)
    oof = pd.concat(predictions, ignore_index=True); oof.to_csv(output / "oof_all.csv", index=False)
    bootstrap_input = oof.rename(columns={"p_uci_only": "p_clinical"})
    bootstrap_variants = ("clinical",) + tuple(name for name in config.variants if name != "uci_only")
    bootstrap = _patient_bootstrap(bootstrap_input, bootstrap_variants, config.seed, config.bootstrap)
    bootstrap.to_csv(output / "patient_bootstrap.csv", index=False)
    candidate_pairs = [
        ("longitudinal_mtl", "shuffled_mtl"),
        ("aligned_mtl", "aligned_shuffled_mtl"),
    ]
    available_pairs = [pair for pair in candidate_pairs if all(name in config.variants for name in pair)]
    external_signal = _paired_patient_bootstrap(
        oof, available_pairs, config.seed, config.bootstrap,
    ) if available_pairs else pd.DataFrame()
    external_signal.to_csv(output / "external_signal_bootstrap.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "config": asdict(config), "longitudinal_external_validation": external_info,
        "shuffled_external_validation": shuffled_info,
        "pair_year_counts": {str(int(k)): int(v) for k, v in pd.Series(years).value_counts().sort_index().items()},
    }, indent=2), encoding="utf-8")
    return summary, bootstrap, external_signal
