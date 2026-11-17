
import argparse
import copy
import json
import random
from pathlib import Path
from _paths import MEPS_BRIDGE, UCI_RAW, result_dir

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

from run_cdc_xgb import load_hospital, make_features


RACES = [
    "Caucasian", "AfricanAmerican",
    "Hispanic", "Asian", "Other",
]

BRIDGE = [
    "age", "male", "high_bp", "high_chol",
    "stroke", "heart_disease", "mi",
] + ["race_" + r for r in RACES]

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

DIAG_GROUPS = {
    "clinical_hypertension": (401, 405),
    "clinical_lipid_disorder": (272, 272),
    "clinical_ischemic_heart": (410, 414),
    "clinical_heart_failure": (428, 428),
    "clinical_cerebrovascular": (430, 438),
    "clinical_renal_disorder": (580, 589),
    "clinical_copd_related": (490, 496),
    "clinical_liver_disorder": (570, 573),
    "clinical_neoplasm": (140, 208),
}

EXPERIMENTS = [
    "Clinical_XGB",
    "UCI_MLP",
    "MEPS_Pretrained",
    "MEPS_Multitask",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diag_flag(df, low, high):
    flags = []

    for col in ["diag_1", "diag_2", "diag_3"]:
        x = (
            df[col].astype("string")
            .str.strip()
            .str.extract(
                r"^(\d{3})(?:\.|$)",
                expand=False,
            )
        )

        x = pd.to_numeric(x, errors="coerce")

        flags.append(
            x.between(low, high)
            .fillna(False)
            .to_numpy(dtype=bool)
        )

    return np.column_stack(flags).any(axis=1)


def add_clinical(df):
    df = df.copy()
    cols = []

    for name, (lo, hi) in DIAG_GROUPS.items():
        df[name] = diag_flag(
            df, lo, hi
        ).astype("int8")
        cols.append(name)

    comp = np.zeros(len(df), dtype=bool)

    for col in ["diag_1", "diag_2", "diag_3"]:
        comp |= (
            df[col].astype("string")
            .str.strip()
            .str.match(
                r"^250\.[1-9][0-9]?$",
                na=False,
            )
            .to_numpy(dtype=bool)
        )

    df["clinical_diabetes_complication"] = (
        comp.astype("int8")
    )
    cols.append("clinical_diabetes_complication")

    df["clinical_comorbidity_count"] = (
        df[cols].sum(axis=1)
        .astype("float32")
    )
    cols.append("clinical_comorbidity_count")

    diag = (
        df[["diag_1", "diag_2", "diag_3"]]
        .astype("string")
        .replace({"?": pd.NA, "": pd.NA})
    )

    df["clinical_unique_diagnoses"] = (
        diag.nunique(axis=1)
        .astype("float32")
    )
    cols.append("clinical_unique_diagnoses")

    present = [x for x in MEDICATIONS if x in df]

    if not present:
        raise ValueError("Medication columns not found")

    meds = df[present].fillna("No").astype(str)

    med_features = {
        "clinical_active_diabetes_drugs":
            meds.isin(["Steady", "Up", "Down"])
            .sum(axis=1),
        "clinical_drugs_increased":
            meds.eq("Up").sum(axis=1),
        "clinical_drugs_decreased":
            meds.eq("Down").sum(axis=1),
        "clinical_drugs_adjusted":
            meds.isin(["Up", "Down"]).sum(axis=1),
    }

    for name, value in med_features.items():
        df[name] = value.astype("float32")
        cols.append(name)

    util = pd.DataFrame(index=df.index)

    for col in [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]:
        util[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0).clip(lower=0)
        )

        name = "clinical_log_" + col

        df[name] = np.log1p(
            util[col]
        ).astype("float32")

        cols.append(name)

    df["clinical_total_prior_visits"] = (
        util.sum(axis=1).astype("float32")
    )

    df["clinical_prior_inpatient_flag"] = (
        util.number_inpatient > 0
    ).astype("int8")

    df["clinical_repeated_inpatient_flag"] = (
        util.number_inpatient >= 2
    ).astype("int8")

    df["clinical_prior_emergency_flag"] = (
        util.number_emergency > 0
    ).astype("int8")

    cols.extend([
        "clinical_total_prior_visits",
        "clinical_prior_inpatient_flag",
        "clinical_repeated_inpatient_flag",
        "clinical_prior_emergency_flag",
    ])

    df["clinical_comorbidity_x_prior_visits"] = (
        df["clinical_comorbidity_count"]
        * np.log1p(df["clinical_total_prior_visits"])
    ).astype("float32")

    cols.append("clinical_comorbidity_x_prior_visits")

    print(f"[Clinical] {len(cols)} features", flush=True)
    return df, cols


def make_bridge(
    age, sex, bp, chol, stroke, heart, mi, race
):
    fields = {
        "age": pd.to_numeric(
            age, errors="coerce"
        ) / 100.0,
        "male": pd.to_numeric(
            sex, errors="coerce"
        ),
        "high_bp": pd.to_numeric(
            bp, errors="coerce"
        ),
        "high_chol": pd.to_numeric(
            chol, errors="coerce"
        ),
        "stroke": pd.to_numeric(
            stroke, errors="coerce"
        ),
        "heart_disease": pd.to_numeric(
            heart, errors="coerce"
        ),
        "mi": pd.to_numeric(
            mi, errors="coerce"
        ),
    }

    result = pd.DataFrame(fields).reset_index(drop=True)

    race = race.reset_index(drop=True)
    race = race.astype("string").str.strip()

    for r in RACES:
        result["race_" + r] = (
            race == r
        ).fillna(False).astype("float32")

    result["age"] = result["age"].where(
        result["age"].between(0, 1)
    )

    for col in BRIDGE[1:7]:
        result[col] = result[col].where(
            result[col].isin([0, 1])
        )

    # A separate missing-value flag could be used;
    # here unknown binary values become -1.
    result["age"] = result["age"].fillna(0.5)

    for col in BRIDGE[1:7]:
        result[col] = result[col].fillna(-1)

    return result[BRIDGE].to_numpy(
        dtype=np.float32
    )


def load_data(hospital_path, meps_path):
    hospital = load_hospital(
        hospital_path
    ).reset_index(drop=True)

    hospital, clinical = add_clinical(hospital)

    stroke = diag_flag(hospital, 430, 438)
    heart = diag_flag(hospital, 410, 414)

    mi = (
        diag_flag(hospital, 410, 410)
        | diag_flag(hospital, 412, 412)
    )

    hb = make_bridge(
        hospital["bridge_age"],
        hospital["bridge_sex"],
        hospital["bridge_high_bp"],
        hospital["bridge_high_chol"],
        pd.Series(stroke),
        pd.Series(heart),
        pd.Series(mi),
        hospital["race"],
    )

    meps = pd.read_csv(
        meps_path,
        low_memory=False,
        dtype={"person_key": str},
    )

    required = [
        "person_key", "age", "sex",
        "high_bp", "high_chol", "stroke",
        "heart_disease", "mi", "race_common",
        "er_any", "inpatient_any",
    ]

    missing = [c for c in required if c not in meps]
    if missing:
        raise ValueError(
            f"Missing MEPS columns: {missing}"
        )

    for col in required:
        if col not in ["person_key", "race_common"]:
            meps[col] = pd.to_numeric(
                meps[col], errors="coerce"
            )

    meps_sex = meps["sex"].map({
        1: 1.0, 2: 0.0
    })

    mb = make_bridge(
        meps["age"],
        meps_sex,
        meps["high_bp"],
        meps["high_chol"],
        meps["stroke"],
        meps["heart_disease"],
        meps["mi"],
        meps["race_common"],
    )

    y_meps = meps[
        ["er_any", "inpatient_any"]
    ].to_numpy(dtype=np.float32)

    valid = np.isfinite(y_meps).all(axis=1)
    valid &= np.isin(y_meps, [0, 1]).all(axis=1)

    mb = mb[valid]
    y_meps = y_meps[valid]

    meps_groups = (
        meps["person_key"]
        .to_numpy()[valid]
    )

    # Explicitly exclude generated bridge features
    # from the private Hospital input.
    clinical_set = set(clinical)
    excluded = {
        "encounter_id", "patient_nbr",
        "readmitted", "target",
    } | clinical_set

    original = [
        c for c in hospital.columns
        if (
            c not in excluded
            and not c.startswith("bridge_")
            and not c.startswith("meps_")
        )
    ]

    feature_cols = original + clinical

    print(
        f"[Hospital] {len(hospital):,} rows; "
        f"{hospital.patient_nbr.nunique():,} patients; "
        f"readmission={hospital.target.mean():.4f}",
        flush=True,
    )

    print(
        f"[MEPS] {len(mb):,} rows; "
        f"{len(np.unique(meps_groups)):,} people; "
        f"ER={y_meps[:, 0].mean():.4f}; "
        f"IP={y_meps[:, 1].mean():.4f}",
        flush=True,
    )

    print(
        f"[Inputs] {len(feature_cols)} Hospital "
        f"source columns; {hb.shape[1]} bridge columns",
        flush=True,
    )

    return (
        hospital, clinical, feature_cols,
        hb, mb, y_meps, meps_groups,
    )


class MultitaskNet(nn.Module):
    def __init__(self, hospital_dim, bridge_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(bridge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.hospital_encoder = nn.Sequential(
            nn.Linear(hospital_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.readmission = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

        self.er_head = nn.Linear(16, 1)
        self.ip_head = nn.Linear(16, 1)

    def hospital_logits(self, x, bridge):
        h = self.hospital_encoder(x)
        z = self.encoder(bridge)

        return self.readmission(
            torch.cat([h, z], dim=1)
        ).squeeze(1)

    def meps_logits(self, bridge):
        z = self.encoder(bridge)
        return torch.stack([
            self.er_head(z).squeeze(1),
            self.ip_head(z).squeeze(1),
        ], dim=1)


def as_tensor(x, device):
    return torch.as_tensor(
        x, dtype=torch.float32, device=device
    )


@torch.no_grad()
def predict_hospital(net, x, bridge, device):
    net.eval()
    output = []

    for start in range(0, len(x), 4096):
        stop = start + 4096

        a = as_tensor(x[start:stop], device)
        b = as_tensor(bridge[start:stop], device)

        p = torch.sigmoid(
            net.hospital_logits(a, b)
        )

        output.append(p.cpu().numpy())

    return np.concatenate(output)


def prepare_matrix(train, val, test, cols):
    matrices = make_features(
        train, val, test, cols
    )

    out = []

    for m in matrices:
        if hasattr(m, "replace"):
            m = m.replace(
                [np.inf, -np.inf], np.nan
            )

        if hasattr(m, "to_numpy"):
            m = m.to_numpy()

        if hasattr(m, "toarray"):
            m = m.toarray()

        m = np.asarray(m, dtype=np.float32)

        m = np.nan_to_num(
            m, nan=0.0, posinf=0.0, neginf=0.0
        )

        out.append(m)

    if not (
        out[0].shape[1]
        == out[1].shape[1]
        == out[2].shape[1]
    ):
        raise ValueError(
            "Hospital feature dimensions differ"
        )

    return out


def pretrain_meps(
    bridge, labels, groups,
    args, device, outdir
):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=args.seed,
    )

    tr, va = next(
        splitter.split(bridge, groups=groups)
    )

    assert set(groups[tr]).isdisjoint(
        set(groups[va])
    )

    # Pretrain only the shared encoder and ER/IP heads.
    set_seed(args.seed + 100)
    net = MultitaskNet(1, bridge.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        list(net.encoder.parameters())
        + list(net.er_head.parameters())
        + list(net.ip_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    xtr = as_tensor(bridge[tr], device)
    ytr = as_tensor(labels[tr], device)

    xva = as_tensor(bridge[va], device)
    yva = as_tensor(labels[va], device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    wait = 0

    print(
        "\n========== MEPS PRETRAINING ==========",
        flush=True,
    )

    for epoch in range(1, args.pretrain_epochs + 1):
        net.train()
        order = torch.randperm(
            len(xtr), device=device
        )

        for idx in order.split(args.meps_batch):
            optimizer.zero_grad()

            logits = net.meps_logits(xtr[idx])

            loss = loss_fn(
                logits, ytr[idx]
            )

            loss.backward()
            optimizer.step()

        net.eval()

        with torch.no_grad():
            val_logits = net.meps_logits(xva)
            val_loss = loss_fn(
                val_logits, yva
            ).item()

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                net.encoder.state_dict()
            )
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[MEPS] epoch={epoch:02d} "
                f"val_loss={val_loss:.5f}",
                flush=True,
            )

        if wait >= args.patience:
            break

    # External held-out group performance is diagnostic.
    net.encoder.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        scores = torch.sigmoid(
            net.meps_logits(xva)
        ).cpu().numpy()

    print(
        f"[MEPS] Best epoch={best_epoch}, "
        f"loss={best_loss:.5f}",
        flush=True,
    )

    for i, name in enumerate(["ER", "IP"]):
        print(
            f"[MEPS] {name} "
            f"AP={average_precision_score(labels[va, i], scores[:, i]):.5f} "
            f"AUROC={roc_auc_score(labels[va, i], scores[:, i]):.5f}",
            flush=True,
        )

    # Refit the external encoder on all MEPS records,
    # using the epoch count selected above.
    set_seed(args.seed + 100)

    final_net = MultitaskNet(
        1, bridge.shape[1]
    ).to(device)

    final_opt = torch.optim.AdamW(
        list(final_net.encoder.parameters())
        + list(final_net.er_head.parameters())
        + list(final_net.ip_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    xa = as_tensor(bridge, device)
    ya = as_tensor(labels, device)

    for _ in range(best_epoch):
        final_net.train()
        order = torch.randperm(
            len(xa), device=device
        )

        for idx in order.split(args.meps_batch):
            final_opt.zero_grad()
            loss = loss_fn(
                final_net.meps_logits(xa[idx]),
                ya[idx],
            )
            loss.backward()
            final_opt.step()

    state = copy.deepcopy(
        final_net.encoder.state_dict()
    )

    torch.save(
        state,
        outdir / "meps_pretrained_encoder.pt",
    )

    print(
        "[MEPS] Encoder saved",
        flush=True,
    )

    return state


def train_net(
    kind, x, bridge, y,
    meps_x, meps_y,
    pretrained_state,
    args, device, seed,
    epochs,
    val_data=None,
):
    set_seed(seed)

    net = MultitaskNet(
        x.shape[1], bridge.shape[1]
    ).to(device)

    if kind == "MEPS_Pretrained":
        net.encoder.load_state_dict(
            pretrained_state
        )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    xt = as_tensor(x, device)
    bt = as_tensor(bridge, device)
    yt = as_tensor(y, device)

    if kind == "MEPS_Multitask":
        xm = as_tensor(meps_x, device)
        ym = as_tensor(meps_y, device)
    else:
        xm = ym = None

    best_ap = -1.0
    best_epoch = 1
    best_state = None
    wait = 0

    rng = np.random.default_rng(seed + 500)

    for epoch in range(1, epochs + 1):
        net.train()

        order = torch.randperm(
            len(xt), device=device
        )

        for indices in order.split(args.batch):
            optimizer.zero_grad()

            pred = net.hospital_logits(
                xt[indices], bt[indices]
            )

            loss = loss_fn(
                pred, yt[indices]
            )

            if kind == "MEPS_Multitask":
                # Draw a fresh MEPS auxiliary batch.
                picked = rng.choice(
                    len(xm),
                    size=min(args.meps_batch, len(xm)),
                    replace=False,
                )

                mi = torch.as_tensor(
                    picked, device=device
                )

                aux = loss_fn(
                    net.meps_logits(xm[mi]),
                    ym[mi],
                )

                loss = (
                    loss + args.lambda_meps * aux
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 5.0
            )

            optimizer.step()

        if val_data is not None:
            xv, bv, yv = val_data

            prob = predict_hospital(
                net, xv, bv, device
            )

            ap = average_precision_score(
                yv, prob
            )

            if ap > best_ap + 1e-5:
                best_ap = ap
                best_epoch = epoch
                best_state = copy.deepcopy(
                    net.state_dict()
                )
                wait = 0
            else:
                wait += 1

            if wait >= args.patience:
                break

    if val_data is not None:
        net.load_state_dict(best_state)

    return net, best_epoch, best_ap


def best_threshold(y, p):
    thresholds = np.arange(
        0.05, 0.6001, 0.005
    )

    values = [
        f1_score(
            y, p >= t, zero_division=0
        )
        for t in thresholds
    ]

    return float(
        thresholds[np.argmax(values)]
    )


def xgb_model(seed):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
    )


def run_cv(
    hospital, feature_cols, hospital_bridge,
    meps_bridge, meps_labels, pretrained,
    args, device, outdir,
):
    cv = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    results = []
    predictions = []

    for fold, (tr, te) in enumerate(
        cv.split(
            hospital,
            hospital.target,
            groups=hospital.patient_nbr,
        ),
        start=1,
    ):
        full_train = hospital.iloc[tr]
        test = hospital.iloc[te]

        assert set(
            full_train.patient_nbr
        ).isdisjoint(
            set(test.patient_nbr)
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=args.seed + fold,
        )

        inner_idx, val_idx = next(
            splitter.split(
                full_train,
                groups=full_train.patient_nbr,
            )
        )

        inner_ids = tr[inner_idx]
        val_ids = tr[val_idx]

        inner = hospital.iloc[inner_ids]
        val = hospital.iloc[val_ids]

        if any(
            p.target.nunique() != 2
            for p in [
                inner, val, full_train, test
            ]
        ):
            raise ValueError(
                f"One-class fold {fold}"
            )

        xi, xv, _ = prepare_matrix(
            inner, val, val, feature_cols
        )

        xf, _, xte = prepare_matrix(
            full_train,
            full_train,
            test,
            feature_cols,
        )

        bi = hospital_bridge[inner_ids]
        bv = hospital_bridge[val_ids]
        bf = hospital_bridge[tr]
        bt = hospital_bridge[te]

        yi = inner.target.to_numpy(
            dtype=np.float32
        )

        yv = val.target.to_numpy(
            dtype=np.float32
        )

        yf = full_train.target.to_numpy(
            dtype=np.float32
        )

        yt = test.target.to_numpy(
            dtype=np.int32
        )

        oof = test[
            ["encounter_id", "patient_nbr", "target"]
        ].copy().rename(
            columns={"target": "y_true"}
        )

        oof["Fold"] = fold

        print(
            f"\n========== FOLD {fold}/{args.folds} ==========",
            flush=True,
        )

        print(
            f"[Split] Train={len(tr):,} "
            f"Inner={len(inner_ids):,} "
            f"Val={len(val_ids):,} "
            f"Test={len(te):,} "
            f"features={xf.shape[1]}",
            flush=True,
        )

        for experiment in EXPERIMENTS:
            if experiment == "Clinical_XGB":
                temp = xgb_model(
                    args.seed + fold
                )
                temp.fit(xi, yi.astype(int))

                vp = temp.predict_proba(
                    xv
                )[:, 1]

                threshold = best_threshold(
                    yv, vp
                )

                final = xgb_model(
                    args.seed + fold
                )
                final.fit(
                    xf, yf.astype(int)
                )

                pred = final.predict_proba(
                    xte
                )[:, 1]

                epochs = np.nan

            else:
                _, epochs, val_ap = train_net(
                    experiment,
                    xi, bi, yi,
                    meps_bridge, meps_labels,
                    pretrained,
                    args, device,
                    args.seed + fold,
                    args.epochs,
                    val_data=(xv, bv, yv),
                )

                # Validation prediction from a fitted
                # inner model, for threshold selection.
                val_model, _, _ = train_net(
                    experiment,
                    xi, bi, yi,
                    meps_bridge, meps_labels,
                    pretrained,
                    args, device,
                    args.seed + fold,
                    epochs,
                )

                vp = predict_hospital(
                    val_model, xv, bv, device
                )

                threshold = best_threshold(
                    yv, vp
                )

                # Refit on all outer training patients.
                final, _, _ = train_net(
                    experiment,
                    xf, bf, yf,
                    meps_bridge, meps_labels,
                    pretrained,
                    args, device,
                    args.seed + fold,
                    epochs,
                )

                pred = predict_hospital(
                    final, xte, bt, device
                )

            row = {
                "Fold": fold,
                "Experiment": experiment,
                "AP": average_precision_score(
                    yt, pred
                ),
                "AUROC": roc_auc_score(
                    yt, pred
                ),
                "F1": f1_score(
                    yt,
                    pred >= threshold,
                    zero_division=0,
                ),
                "Brier": brier_score_loss(
                    yt, pred
                ),
                "Threshold": threshold,
                "Epochs": epochs,
            }

            results.append(row)

            oof["p_" + experiment] = pred

            print(
                f"[Done] Fold={fold} "
                f"{experiment:19s} "
                f"AP={row['AP']:.5f} "
                f"AUROC={row['AUROC']:.5f} "
                f"F1={row['F1']:.5f} "
                f"epochs={epochs}",
                flush=True,
            )

        predictions.append(oof)

        pd.DataFrame(results).to_csv(
            outdir / "partial_results.csv",
            index=False,
        )

    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True),
    )


def paired_bootstrap(oof, args, outdir):
    pairs = [
        ("MEPS_Pretrained", "UCI_MLP"),
        ("MEPS_Multitask", "UCI_MLP"),
        ("MEPS_Pretrained", "Clinical_XGB"),
        ("MEPS_Multitask", "Clinical_XGB"),
    ]

    patients, inv = np.unique(
        oof.patient_nbr.to_numpy(),
        return_inverse=True,
    )

    y = oof.y_true.to_numpy()
    rng = np.random.default_rng(args.seed + 999)

    rows = []

    for current, reference in pairs:
        a = oof["p_" + current].to_numpy()
        b = oof["p_" + reference].to_numpy()

        observed = (
            average_precision_score(y, a)
            - average_precision_score(y, b)
        )

        deltas = []

        for _ in range(args.bootstrap):
            drawn = rng.integers(
                0, len(patients),
                size=len(patients),
            )

            counts = np.bincount(
                drawn, minlength=len(patients)
            )

            weights = counts[inv]

            d = (
                average_precision_score(
                    y, a, sample_weight=weights
                )
                - average_precision_score(
                    y, b, sample_weight=weights
                )
            )

            deltas.append(d)

        rows.append({
            "Comparison": current + " - " + reference,
            "DeltaAP": observed,
            "CI95_Lower": np.quantile(
                deltas, 0.025
            ),
            "CI95_Upper": np.quantile(
                deltas, 0.975
            ),
        })

    table = pd.DataFrame(rows)

    table.to_csv(
        outdir / "patient_bootstrap.csv",
        index=False,
    )

    print(
        "\n========== PATIENT BOOTSTRAP ==========",
        flush=True,
    )
    print(
        table.round(6).to_string(index=False),
        flush=True,
    )


def summarize(results, oof, args, outdir):
    results.to_csv(
        outdir / "fold_results.csv",
        index=False,
    )

    oof.to_csv(
        outdir / "oof_all.csv",
        index=False,
    )

    summary = (
        results.groupby(
            "Experiment", sort=False
        )[["AP", "AUROC", "F1", "Brier"]]
        .agg(["mean", "std"])
    )

    summary.columns = [
        a + "_" + b for a, b in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        outdir / "summary.csv",
        index=False,
    )

    print(
        "\n========== FINAL 5-FOLD SUMMARY ==========",
        flush=True,
    )

    print(
        summary.round(6).to_string(index=False),
        flush=True,
    )

    pivot = results.pivot(
        index="Fold",
        columns="Experiment",
        values="AP",
    )

    comparisons = [
        ("MEPS_Pretrained", "UCI_MLP"),
        ("MEPS_Multitask", "UCI_MLP"),
        ("MEPS_Pretrained", "Clinical_XGB"),
        ("MEPS_Multitask", "Clinical_XGB"),
    ]

    rows = []

    for current, reference in comparisons:
        difference = (
            pivot[current] - pivot[reference]
        )

        rows.append({
            "Comparison": current + " - " + reference,
            "MeanDeltaAP": difference.mean(),
            "PositiveFolds": int(
                (difference > 0).sum()
            ),
            "PooledOOF_DeltaAP":
                average_precision_score(
                    oof.y_true,
                    oof["p_" + current],
                )
                - average_precision_score(
                    oof.y_true,
                    oof["p_" + reference],
                ),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        outdir / "paired_comparisons.csv",
        index=False,
    )

    print(
        "\n========== PAIRED COMPARISONS ==========",
        flush=True,
    )
    print(
        comparison.round(6).to_string(index=False),
        flush=True,
    )

    if args.bootstrap > 0:
        paired_bootstrap(oof, args, outdir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hospital", default=UCI_RAW
    )

    parser.add_argument(
        "--meps",
        default=MEPS_BRIDGE,
    )

    parser.add_argument(
        "--outdir",
        default=result_dir("meps_multitask_results"),
    )

    parser.add_argument(
        "--folds", type=int, default=5
    )

    parser.add_argument(
        "--seed", type=int, default=777
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )

    parser.add_argument(
        "--pretrain-epochs", type=int, default=30
    )

    parser.add_argument(
        "--patience", type=int, default=6
    )

    parser.add_argument(
        "--batch", type=int, default=1024
    )

    parser.add_argument(
        "--meps-batch", type=int, default=512
    )

    parser.add_argument(
        "--lambda-meps", type=float, default=0.2
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--bootstrap", type=int, default=200
    )

    args = parser.parse_args()

    if (
        args.folds < 2
        or args.epochs < 1
        or args.pretrain_epochs < 1
        or args.patience < 1
        or args.batch < 1
        or args.meps_batch < 1
        or args.lambda_meps < 0
        or args.bootstrap < 0
    ):
        parser.error("Invalid parameter")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Device] {device}",
        flush=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    (
        hospital, clinical, feature_cols,
        hb, mb, ym, meps_groups
    ) = load_data(
        args.hospital, args.meps
    )

    state = pretrain_meps(
        mb, ym, meps_groups,
        args, device, outdir,
    )

    results, oof = run_cv(
        hospital, feature_cols, hb,
        mb, ym, state,
        args, device, outdir,
    )

    summarize(
        results, oof, args, outdir
    )

    print(
        "\n========== FINISHED ==========",
        flush=True,
    )
    print(
        "Saved:", outdir.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()
