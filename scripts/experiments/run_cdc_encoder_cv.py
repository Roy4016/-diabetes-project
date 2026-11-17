
import argparse
import copy
import json
import random
from pathlib import Path
from _paths import CDC_RAW, UCI_RAW, result_dir

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from run_cdc_xgb import (
    BRIDGE_COLS,
    CDC_AGE_MID,
    load_hospital,
    make_features,
    choose_f1_threshold,
)


TASKS = [
    "poor_health",
    "diff_walk",
    "inactive",
    "smoker",
]

TASK_COLUMNS = [
    "GenHlth",
    "DiffWalk",
    "PhysActivity",
    "Smoker",
]

SCORE_COLS = [
    "cdc_score_" + t for t in TASKS
]

EMBED_COLS = [
    f"cdc_emb_{i:02d}" for i in range(8)
]

EXPERIMENTS = {
    "E1_Bridge": [],
    "E2_CDC_Scores": SCORE_COLS,
    "E3_CDC_Embedding": EMBED_COLS,
    "E4_CDC_Combined": SCORE_COLS + EMBED_COLS,
}

EXCLUDED = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "target",
    *BRIDGE_COLS,
    *SCORE_COLS,
    *EMBED_COLS,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cdc(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [
        "Diabetes_012",
        "Age", "Sex", "HighBP", "HighChol",
        *TASK_COLUMNS,
    ]

    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"CDC missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    # Train CDC encoder on diabetic respondents.
    df = df[df["Diabetes_012"] == 2].copy()

    df["bridge_age"] = df["Age"].map(CDC_AGE_MID)
    df["bridge_sex"] = df["Sex"]
    df["bridge_high_bp"] = df["HighBP"]
    df["bridge_high_chol"] = df["HighChol"]

    df = df.dropna(subset=BRIDGE_COLS)

    for c in BRIDGE_COLS[1:]:
        df = df[df[c].isin([0, 1])]

    for c in TASK_COLUMNS:
        valid = [1, 2, 3, 4, 5] if c == "GenHlth" else [0, 1]
        df = df[df[c].isin(valid)]

    labels = np.column_stack([
        (df["GenHlth"] >= 4).astype(np.float32),
        (df["DiffWalk"] == 1).astype(np.float32),
        (df["PhysActivity"] == 0).astype(np.float32),
        (df["Smoker"] == 1).astype(np.float32),
    ])

    print(
        f"[CDC] Eligible diabetic respondents: {len(df):,}"
    )

    for i, name in enumerate(TASKS):
        print(
            f"  {name}: prevalence={labels[:, i].mean():.4f}"
        )

    return df, labels


def bridge_to_numpy(df):
    X = df[BRIDGE_COLS].copy()

    for c in BRIDGE_COLS:
        X[c] = pd.to_numeric(
            X[c], errors="coerce"
        )

    # CDC-derived features must use the same
    # representation for both source and target.
    # Unknown Hospital bridge values are imputed
    # with source-dataset statistics.
    return X.to_numpy(dtype=np.float32)


class HealthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.head = nn.Linear(8, 4)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return z, logits


def prepare_encoder_input(X, fill_values):
    X = X.copy()
    for j in range(X.shape[1]):
        invalid = ~np.isfinite(X[:, j])
        X[invalid, j] = fill_values[j]

    # Age is the only continuous input.
    X[:, 0] = X[:, 0] / 100.0

    return X.astype(np.float32)


def train_encoder(
    cdc,
    labels,
    seed,
    device,
    epochs,
    outdir,
):
    X_raw = bridge_to_numpy(cdc)

    # Imputation learned exclusively from CDC.
    fill_values = np.nanmedian(X_raw, axis=0)
    fill_values = np.where(
        np.isfinite(fill_values), fill_values, 0
    )

    X = prepare_encoder_input(
        X_raw, fill_values
    )

    indices = np.arange(len(X))

    idx_train, idx_val = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels[:, 0],
    )

    train_ds = TensorDataset(
        torch.from_numpy(X[idx_train]),
        torch.from_numpy(labels[idx_train]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        num_workers=0,
    )

    xval = torch.from_numpy(
        X[idx_val]
    ).to(device)

    yval = torch.from_numpy(
        labels[idx_val]
    ).to(device)

    model = HealthEncoder().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    patience = 10

    print("\n========== CDC PRETRAINING ==========")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            _, logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            _, val_logits = model(xval)
            val_loss = criterion(
                val_logits, yval
            ).item()

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[CDC] Epoch {epoch:03d}: "
                f"train={train_loss:.5f}, "
                f"val={val_loss:.5f}",
                flush=True,
            )

        if bad_epochs >= patience:
            print("[CDC] Early stopping")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z, logits = model(xval)
        p = torch.sigmoid(logits).cpu().numpy()

    print("\n========== CDC VALIDATION ==========")

    for j, task in enumerate(TASKS):
        auc = roc_auc_score(
            labels[idx_val, j], p[:, j]
        )
        ap = average_precision_score(
            labels[idx_val, j], p[:, j]
        )
        print(
            f"{task:14s}: "
            f"AUROC={auc:.4f}, AP={ap:.4f}"
        )

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "fill_values": fill_values,
            "seed": seed,
        },
        outdir / "cdc_health_encoder.pt",
    )

    return model.to(device), fill_values


@torch.no_grad()
def attach_encoder_features(
    hospital, model, fill_values, device
):
    raw = bridge_to_numpy(hospital)
    X = prepare_encoder_input(
        raw, fill_values
    )

    # Batch inference, avoids large device transfers.
    z_parts = []
    p_parts = []

    model.eval()

    for start in range(0, len(X), 8192):
        xb = torch.from_numpy(
            X[start:start + 8192]
        ).to(device)

        z, logits = model(xb)

        z_parts.append(
            z.cpu().numpy()
        )
        p_parts.append(
            torch.sigmoid(logits).cpu().numpy()
        )

    embeddings = np.concatenate(z_parts)
    probabilities = np.concatenate(p_parts)

    for i, col in enumerate(EMBED_COLS):
        hospital[col] = embeddings[:, i]

    for i, col in enumerate(SCORE_COLS):
        hospital[col] = probabilities[:, i]

    print("\n========== TRANSFER TO HOSPITAL ==========")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Score shape: {probabilities.shape}")

    return hospital


def get_columns(df, experiment):
    cols = [
        c for c in df.columns
        if c not in EXCLUDED
    ]

    cols += BRIDGE_COLS
    cols += EXPERIMENTS[experiment]

    return list(dict.fromkeys(cols))


def encode_data(train, val, test, cols):
    xtr, xva, xte = make_features(
        train, val, test, cols
    )

    # Same missing-value treatment across
    # all four experiments.
    xtr = xtr.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)
    xva = xva.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)
    xte = xte.replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)

    medians = xtr.median().fillna(0)

    xtr = xtr.fillna(medians).fillna(0)
    xva = xva.fillna(medians).fillna(0)
    xte = xte.fillna(medians).fillna(0)

    return xtr, xva, xte


def create_xgb(seed, weight):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=weight,
    )


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "Recall": recall_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "Brier": brier_score_loss(y, p),
    }


def run_fold(
    hospital,
    outer_train_idx,
    outer_test_idx,
    fold,
    seed,
    outdir,
):
    outer_train = hospital.iloc[
        outer_train_idx
    ].copy()
    outer_test = hospital.iloc[
        outer_test_idx
    ].copy()

    # Inner patient-disjoint validation is used
    # for class-weight and threshold selection.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed + fold,
    )

    itr, iva = next(
        splitter.split(
            outer_train,
            outer_train["target"],
            groups=outer_train["patient_nbr"],
        )
    )

    inner_train = outer_train.iloc[itr].copy()
    inner_val = outer_train.iloc[iva].copy()

    assert set(
        inner_train["patient_nbr"]
    ).isdisjoint(set(inner_val["patient_nbr"]))

    yi = inner_train["target"].to_numpy()
    yv = inner_val["target"].to_numpy()
    yt = outer_train["target"].to_numpy()
    yh = outer_test["target"].to_numpy()

    for label, y in [
        ("inner_train", yi),
        ("inner_val", yv),
        ("outer_train", yt),
        ("outer_test", yh),
    ]:
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"Fold {fold}: {label} has one class"
            )

    natural_weight = (
        int((yi == 0).sum())
        / max(int((yi == 1).sum()), 1)
    )

    # Duplicate values are removed automatically.
    weights = list(dict.fromkeys([
        1.0, 2.0, 4.0, natural_weight
    ]))

    rows = []

    for experiment in EXPERIMENTS:
        cols = get_columns(
            hospital, experiment
        )

        xi, xv, _ = encode_data(
            inner_train, inner_val,
            inner_val, cols
        )

        best_ap = -np.inf
        best_weight = None
        best_threshold = None

        for weight in weights:
            model = create_xgb(
                seed + fold, weight
            )
            model.fit(xi, yi)

            pval = model.predict_proba(
                xv
            )[:, 1]

            ap = average_precision_score(
                yv, pval
            )

            threshold = choose_f1_threshold(
                yv, pval
            )

            print(
                f"[Inner] Fold={fold} "
                f"{experiment} "
                f"weight={weight:.3f} "
                f"AP={ap:.4f} "
                f"threshold={threshold:.3f}",
                flush=True,
            )

            if ap > best_ap:
                best_ap = ap
                best_weight = weight
                best_threshold = threshold

        # No outer-test information is used for
        # feature encoding or hyperparameter tuning.
        xtr, _, xte = encode_data(
            outer_train, outer_train,
            outer_test, cols
        )

        final_weight = (
            int((yt == 0).sum())
            / max(int((yt == 1).sum()), 1)
            if abs(best_weight - natural_weight) < 1e-8
            else best_weight
        )

        model = create_xgb(
            seed + fold, final_weight
        )
        model.fit(xtr, yt)

        probability = model.predict_proba(
            xte
        )[:, 1]

        result = metrics(
            yh,
            probability,
            best_threshold,
        )

        row = {
            "Fold": fold,
            "Experiment": experiment,
            "Features": xtr.shape[1],
            "SelectedWeight": best_weight,
            "FinalWeight": final_weight,
            "InnerAP": best_ap,
            "Threshold": best_threshold,
            "PositiveRate": yh.mean(),
            **result,
        }
        rows.append(row)

        prediction_df = pd.DataFrame({
            "encounter_id": (
                outer_test["encounter_id"].values
                if "encounter_id" in outer_test
                else outer_test.index.to_numpy()
            ),
            "patient_nbr": outer_test[
                "patient_nbr"
            ].values,
            "y_true": yh,
            "y_prob": probability,
            "y_pred": (
                probability >= best_threshold
            ).astype(int),
        })

        prediction_df.to_csv(
            outdir / (
                f"{experiment}_fold{fold}_pred.csv"
            ),
            index=False,
        )

        print(
            f"[Done] Fold={fold} "
            f"{experiment}: "
            f"weight={final_weight:.3f}, "
            f"AP={result['AP']:.4f}, "
            f"AUROC={result['AUROC']:.4f}, "
            f"F1={result['F1']:.4f}",
            flush=True,
        )

    return rows


def save_summary(results, outdir):
    df = pd.DataFrame(results)

    df.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    cols = [
        "AP", "AUROC", "F1",
        "Precision", "Recall", "Brier"
    ]

    summary = (
        df.groupby("Experiment", sort=False)[cols]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()
    summary.to_csv(
        outdir / "summary_mean_std.csv",
        index=False,
    )

    pivot = df.pivot(
        index="Fold",
        columns="Experiment",
        values="AP",
    )

    baseline = pivot["E1_Bridge"]

    delta_rows = []

    for name in list(EXPERIMENTS)[1:]:
        diff = pivot[name] - baseline

        delta_rows.append({
            "Experiment": name,
            "MeanDeltaAP": diff.mean(),
            "StdDeltaAP": diff.std(),
            "PositiveFolds": int(
                (diff > 0).sum()
            ),
            "TotalFolds": len(diff),
        })

    delta = pd.DataFrame(delta_rows)
    delta.to_csv(
        outdir / "paired_delta_ap.csv",
        index=False,
    )

    print("\n========== FINAL 5-FOLD SUMMARY ==========")
    print(
        summary[
            [
                "Experiment",
                "AP_mean", "AP_std",
                "AUROC_mean", "AUROC_std",
                "F1_mean", "F1_std",
                "Brier_mean",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n========== CDC VS BRIDGE ==========")
    print(
        delta.round(5).to_string(index=False)
    )

    print("\n========== SELECTED WEIGHTS ==========")
    print(
        df.pivot(
            index="Fold",
            columns="Experiment",
            values="SelectedWeight",
        ).round(3).to_string()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital", default=UCI_RAW
    )
    parser.add_argument(
        "--cdc",
        default=CDC_RAW,
    )
    parser.add_argument(
        "--seed", type=int, default=777
    )
    parser.add_argument(
        "--folds", type=int, default=5
    )
    parser.add_argument(
        "--epochs", type=int, default=100
    )
    parser.add_argument(
        "--outdir",
        default=result_dir("cdc_encoder_cv_results"),
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[Device] PyTorch: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        outdir / "config.json",
        "w",
    ) as f:
        json.dump(vars(args), f, indent=2)

    print("\n========== LOAD DATA ==========")

    hospital = load_hospital(args.hospital)
    cdc, labels = load_cdc(args.cdc)

    encoder, fill_values = train_encoder(
        cdc, labels,
        args.seed, device,
        args.epochs, outdir,
    )

    hospital = attach_encoder_features(
        hospital, encoder,
        fill_values, device,
    )

    print("\n========== 5-FOLD CV ==========")

    cv = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    folds = list(
        cv.split(
            hospital,
            hospital["target"],
            groups=hospital["patient_nbr"],
        )
    )

    results = []
    assignments = []

    for fold, (itr, ite) in enumerate(
        folds, start=1
    ):
        train_patients = set(
            hospital.iloc[itr]["patient_nbr"]
        )
        test_patients = set(
            hospital.iloc[ite]["patient_nbr"]
        )

        assert train_patients.isdisjoint(
            test_patients
        )

        print(
            f"\n========== FOLD {fold}/{args.folds} "
            f"Train={len(itr):,} "
            f"Test={len(ite):,} ==========",
            flush=True,
        )

        a = hospital.iloc[ite][
            ["patient_nbr"]
        ].copy()

        if "encounter_id" in hospital:
            a["encounter_id"] = (
                hospital.iloc[ite][
                    "encounter_id"
                ].values
            )

        a["fold"] = fold
        assignments.append(a)

        fold_results = run_fold(
            hospital, itr, ite,
            fold, args.seed, outdir,
        )

        results.extend(fold_results)

        pd.DataFrame(results).to_csv(
            outdir / "partial_results.csv",
            index=False,
        )

    pd.concat(assignments).to_csv(
        outdir / "fold_assignments.csv",
        index=False,
    )

    save_summary(results, outdir)

    print("\n========== FINISHED ==========")
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
