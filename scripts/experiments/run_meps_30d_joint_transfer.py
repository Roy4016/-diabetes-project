#!/usr/bin/env python3
"""Test MEPS 30-day grouping and joint UCI/MEPS training on the fixed five folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.meps_30d_transfer import (  # noqa: E402
    common_matrix_meps, common_matrix_uci, group_support, load_events,
)
from diabetes_project.paper_comparison import (  # noqa: E402
    catboost_shap_sample, make_paper_catboost_features, top_fraction_metrics,
)
from run_dual_external_meps_nhanes import (  # noqa: E402
    align_catboost, hospital_frame, load_reference,
)
from run_paper_fair_comparison import paired_patient_bootstrap  # noqa: E402


class JointNetwork:
    """Small shared bridge encoder with separate UCI and MEPS heads."""

    def __init__(self, common_dim, uci_dim, meps_dim):
        import torch

        nn = torch.nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.common = nn.Sequential(nn.Linear(common_dim, 48), nn.ReLU(),
                                            nn.Linear(48, 24), nn.ReLU())
                self.uci_private = nn.Sequential(nn.Linear(uci_dim, 64), nn.ReLU(), nn.Dropout(.10))
                self.meps_private = nn.Sequential(nn.Linear(meps_dim, 12), nn.ReLU())
                self.uci_head = nn.Sequential(nn.Linear(88, 32), nn.ReLU(), nn.Linear(32, 1))
                self.meps_head = nn.Sequential(nn.Linear(36, 16), nn.ReLU(), nn.Linear(16, 1))

            def uci(self, common, private):
                return self.uci_head(torch.cat([self.common(common), self.uci_private(private)], 1)).squeeze(1)

            def meps(self, common, private):
                return self.meps_head(torch.cat([self.common(common), self.meps_private(private)], 1)).squeeze(1)

        self.model = Model()


def encoded_uci_private(raw, categorical, train_mask):
    numeric = [name for name in raw if name not in categorical]
    processor = ColumnTransformer([
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                      sparse_output=False, dtype=np.float32)),
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ], sparse_threshold=0.)
    processor.fit(raw.loc[train_mask])
    result = processor.transform(raw)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite UCI neural features")
    return result


def fit_joint(common_uci, private_uci, y_uci, common_meps, private_meps, y_meps,
              train_mask, test_mask, args, fold, external):
    import torch

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed + fold)
    network = JointNetwork(common_uci.shape[1], private_uci.shape[1], private_meps.shape[1]).model
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    loss = torch.nn.BCEWithLogitsLoss()
    common_train = torch.from_numpy(common_uci[train_mask].to_numpy(dtype=np.float32))
    private_train = torch.from_numpy(private_uci[train_mask])
    target_train = torch.from_numpy(y_uci[train_mask].astype(np.float32))
    common_source = torch.from_numpy(common_meps.to_numpy(dtype=np.float32))
    private_source = torch.from_numpy(private_meps.to_numpy(dtype=np.float32))
    target_source = torch.from_numpy(y_meps.astype(np.float32))
    rng = np.random.default_rng(args.seed + fold)
    log = []
    for epoch in range(args.epochs):
        network.train()
        order = rng.permutation(len(target_train))
        losses = []
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            ids = order[start:start + args.batch_size]
            prediction = network.uci(common_train[ids], private_train[ids])
            uci_loss = loss(prediction, target_train[ids])
            total = uci_loss
            if external and step % args.meps_interval == 0:
                source_ids = rng.integers(0, len(target_source), size=min(args.batch_size, len(target_source)))
                source_prediction = network.meps(common_source[source_ids], private_source[source_ids])
                total = total + args.external_weight * loss(source_prediction, target_source[source_ids])
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            losses.append(float(uci_loss.detach()))
        log.append({"Fold": fold, "Model": "joint_meps_30d" if external else "uci_only_nn",
                    "Epoch": epoch + 1, "UCI_Train_Loss": float(np.mean(losses))})
    network.eval()
    with torch.no_grad():
        common_test = torch.from_numpy(common_uci[test_mask].to_numpy(dtype=np.float32))
        private_test = torch.from_numpy(private_uci[test_mask])
        result = torch.sigmoid(network.uci(common_test, private_test)).numpy()
    return result, log


def fit_meps_teacher(common_meps, targets, persons, common_uci, args):
    from catboost import CatBoostClassifier

    # The teacher sees only features that have the same meaning at UCI inference.
    features = common_meps.reset_index(drop=True)
    train, validation = next(GroupShuffleSplit(test_size=.20, random_state=args.seed).split(
        features, targets, groups=persons))
    if set(persons[train]) & set(persons[validation]):
        raise ValueError("MEPS teacher validation has patient overlap")

    def model():
        return CatBoostClassifier(iterations=args.teacher_iterations, depth=4,
                                  learning_rate=.035, l2_leaf_reg=10, loss_function="Logloss",
                                  random_seed=args.seed, thread_count=args.threads,
                                  allow_writing_files=False, verbose=False)

    probe = model()
    probe.fit(features.iloc[train], targets[train])
    held = probe.predict_proba(features.iloc[validation])[:, 1]
    diagnostic = {"MEPS_Validation_Rows": int(len(validation)),
                  "MEPS_Validation_Patients": int(len(set(persons[validation]))),
                  "MEPS_Validation_PR_AUC": float(average_precision_score(targets[validation], held)),
                  "MEPS_Validation_AUROC": float(roc_auc_score(targets[validation], held))}
    teacher = model()
    teacher.fit(features, targets)
    risk = teacher.predict_proba(common_uci[features.columns])[:, 1]
    return risk, diagnostic


def fit_group_catboost(raw, categorical, risk, support, targets, train_mask, test_mask, args):
    from catboost import CatBoostClassifier, Pool

    features = raw.copy()
    features["meps_30d_teacher_risk"] = risk
    features["meps_ccs_log_support"] = support
    model = CatBoostClassifier(iterations=args.catboost_iterations, depth=6,
                               learning_rate=.035, l2_leaf_reg=15, loss_function="Logloss",
                               random_seed=2026, thread_count=args.threads,
                               allow_writing_files=False, verbose=False)
    model.fit(Pool(features.loc[train_mask], targets[train_mask], cat_features=categorical))
    test_features = features.loc[test_mask]
    probability = model.predict_proba(Pool(test_features, cat_features=categorical))[:, 1]
    return model, test_features, probability


def summarize(oof, names, args):
    records = []
    for fold, part in oof.groupby("Fold", sort=True):
        for name in names:
            records.append({"Fold": int(fold), "Model": name,
                            "Rows": len(part), "Patients": int(part.patient_nbr.nunique()),
                            **top_fraction_metrics(part.y_true, part[name], args.alert_fraction)})
    folds = pd.DataFrame(records)
    folds.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary = folds.groupby("Model", sort=False).agg(
        Folds=("Fold", "size"), PR_AUC_mean=("PR_AUC", "mean"), PR_AUC_std=("PR_AUC", "std"),
        AUROC_mean=("AUROC", "mean"), Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"), F1_mean=("F1_score", "mean"),
        Brier_mean=("Brier_Score", "mean"),
    ).reset_index()
    summary["Pooled_PR_AUC"] = [average_precision_score(oof.y_true, oof[name]) for name in summary.Model]
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(summary.round(6).to_string(index=False), flush=True)
    comparisons = [
        ("group_teacher_catboost", "catboost_raw"),
        ("group_teacher_catboost_hgb50", "ccs_only_hgb_catboost50"),
        ("joint_meps_30d", "uci_only_nn"),
        ("joint_meps_30d_catboost50", "uci_only_nn_catboost50"),
        ("joint_meps_30d_catboost50", "ccs_only_hgb_catboost50"),
        ("group_teacher_catboost_hgb50", "ccs_only_hgb_catboost50"),
    ]
    paired_patient_bootstrap(oof, comparisons, args.outdir, args.bootstrap, args.seed)


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference_oof)
    catboost_raw = align_catboost(reference, args.catboost_oof)
    hospital, hospital_columns = hospital_frame(args, reference)
    events, linked, groups = load_events(args.event_data, args.event_ccs, args.top_groups)
    common_uci = common_matrix_uci(hospital, groups)
    common_meps, private_meps = common_matrix_meps(events, linked, groups)
    support = group_support(common_uci, common_meps, groups)
    targets = reference.y_true.to_numpy(dtype=int)
    source_targets = events.readmission_30d.to_numpy(dtype=int)
    source_persons = events.person_key.to_numpy()
    risk, diagnostic = fit_meps_teacher(common_meps, source_targets,
                                        source_persons, common_uci, args)
    raw, categorical = make_paper_catboost_features(
        hospital[["encounter_id", "patient_nbr", "target", *hospital_columns[:46]]], raw_icd=True)
    output = reference[["encounter_id", "patient_nbr", "y_true", "Fold", "p_ccs_only_hgb"]].copy()
    output = output.rename(columns={"p_ccs_only_hgb": "ccs_only_hgb"})
    output["catboost_raw"] = catboost_raw
    output["ccs_only_hgb_catboost50"] = (output.ccs_only_hgb + output.catboost_raw) / 2
    output["meps_teacher_risk"] = risk
    output["meps_ccs_log_support"] = support
    logs = []
    shap_tables = []
    folds = list(range(1, args.max_folds + 1))
    for fold in folds:
        test = output.Fold.eq(fold).to_numpy()
        train = ~test
        private_uci = encoded_uci_private(raw, categorical, train)
        for external, name in ((False, "uci_only_nn"), (True, "joint_meps_30d")):
            prediction, training = fit_joint(common_uci, private_uci, targets,
                                             common_meps, private_meps, source_targets,
                                             train, test, args, fold, external)
            output.loc[test, name] = prediction
            output.loc[test, f"{name}_catboost50"] = (prediction + catboost_raw[test]) / 2
            logs.extend(training)
            print(f"[Fold {fold}] {name} AP={average_precision_score(targets[test], prediction):.6f}",
                  flush=True)
        model, catboost_test, prediction = fit_group_catboost(
            raw, categorical, risk, support, targets, train, test, args)
        output.loc[test, "group_teacher_catboost"] = prediction
        output.loc[test, "group_teacher_catboost_hgb50"] = (prediction + output.loc[test, "ccs_only_hgb"]) / 2
        if args.shap_sample:
            shap = catboost_shap_sample(model, catboost_test, categorical,
                                       args.shap_sample, args.seed + fold)
            shap.insert(0, "Fold", fold)
            shap_tables.append(shap)
        print(f"[Fold {fold}] group_teacher_catboost AP={average_precision_score(targets[test], prediction):.6f}",
              flush=True)
        output.loc[test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)
    names = ["ccs_only_hgb", "catboost_raw", "ccs_only_hgb_catboost50",
             "uci_only_nn", "joint_meps_30d", "uci_only_nn_catboost50",
             "joint_meps_30d_catboost50", "group_teacher_catboost",
             "group_teacher_catboost_hgb50"]
    completed = output.loc[output.Fold.isin(folds)].copy()
    if completed[names].isna().any().any():
        raise ValueError("Incomplete outer-fold predictions")
    completed.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    pd.DataFrame(logs).to_csv(args.outdir / "training.csv", index=False)
    if shap_tables:
        shap_fold = pd.concat(shap_tables, ignore_index=True)
        shap_fold.to_csv(args.outdir / "shap_fold.csv", index=False)
        summary = pd.DataFrame([{
            "Feature": feature,
            "MeanAbsSHAP_logodds": float(np.average(part.MeanAbsSHAP_logodds,
                                                      weights=part.Explained_Rows)),
            "MeanSHAP_logodds": float(np.average(part.MeanSHAP_logodds,
                                                   weights=part.Explained_Rows)),
            "Explained_Rows": int(part.Explained_Rows.sum()),
        } for feature, part in shap_fold.groupby("Feature", sort=False)])
        summary = summary.sort_values("MeanAbsSHAP_logodds", ascending=False)
        summary["Rank"] = np.arange(1, len(summary) + 1)
        summary.to_csv(args.outdir / "shap_global.csv", index=False)
    summarize(completed, names, args)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"completed_folds": folds, "meps_events": len(events),
                   "meps_people": int(events.person_key.nunique()),
                   "meps_positive": int(source_targets.sum()), "ccs_groups": groups,
                   "meps_teacher_diagnostic": diagnostic,
                   "split": "fixed patient-disjoint five-fold seed-777 reference OOF",
                   "external_label": "MEPS exact-date 30-day inpatient readmission",
                   "uci_label": "UCI 30-day inpatient readmission",
                   "catboost_selection": "fixed depth=6, 750 trees from prior train-only selection",
                   "threshold": "top fraction within each outer test fold"})
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--catboost-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--event-data", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/meps_30d_joint_transfer_seed777")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--top-groups", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--external-weight", type=float, default=.3)
    parser.add_argument("--meps-interval", type=int, default=4)
    parser.add_argument("--teacher-iterations", type=int, default=400)
    parser.add_argument("--catboost-iterations", type=int, default=750)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--shap-sample", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5 or args.top_groups < 1 or args.epochs < 1 or args.batch_size < 1 or args.external_weight < 0 or args.meps_interval < 1 or args.bootstrap < 0 or args.shap_sample < 0:
        parser.error("Invalid fold count or hyperparameters")
    for path in (args.hospital, args.ccs_file, args.reference_oof, args.catboost_oof,
                 args.event_data, args.event_ccs):
        if not path.is_file():
            parser.error(f"Required file missing: {path}")
    run(args)


if __name__ == "__main__":
    main()
