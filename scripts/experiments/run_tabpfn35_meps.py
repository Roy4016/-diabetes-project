#!/usr/bin/env python3
"""Evaluate local TabPFN 3.5 with and without real MEPS event-CCS features."""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diabetes_project.paper_comparison import top_fraction_metrics

from _paths import CCS_RAW, UCI_RAW, result_dir
from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_ccs_group_hgb import (
    DIAG, EVENT_STATS, attach_group_features, attach_uci_ccs, make_event_group_rates,
)
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs


VARIANTS = {
    "tabpfn35_ccs_only": [],
    "tabpfn35_ccs_meps": list(EVENT_STATS),
}
CATEGORICAL_IDS = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
CONTEXT_HGB_PARAMS = {
    "learning_rate": 0.05, "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 20, "l2_regularization": 1.0,
}


def tabpfn_frame(frame, columns):
    """Preserve raw tabular values; declare category-like IDs as strings."""
    output = frame.loc[:, columns].copy()
    categorical = []
    for index, name in enumerate(columns):
        series = output[name]
        if (name in CATEGORICAL_IDS or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)):
            output[name] = series.astype("string").fillna("Missing").astype(str)
            categorical.append(index)
        else:
            output[name] = pd.to_numeric(series, errors="coerce").astype(np.float32)
    return output, categorical


def get_tabpfn(model_version):
    try:
        installed = version("tabpfn")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError(
            "TabPFN is not installed. Run: python -m pip install -e '.[tabpfn]'"
        ) from exc
    selected = "V3_5_FAST" if model_version == "3.5-fast" else "V3_5"
    if not hasattr(ModelVersion, selected):
        raise RuntimeError(f"Installed tabpfn {installed} lacks TabPFN 3.5; update tabpfn")
    return TabPFNClassifier, getattr(ModelVersion, selected), installed


def aligned_reference(path, hospital):
    reference = pd.read_csv(path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_hgb_tuned", "p_ccs_only_hgb",
    ])
    if reference.encounter_id.duplicated().any() or hospital.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs")
    aligned = reference.set_index("encounter_id").reindex(hospital.encounter_id.to_numpy())
    if len(aligned) != len(hospital) or aligned.isna().any().any():
        raise ValueError("Reference OOF does not cover hospital data")
    if not np.array_equal(aligned.patient_nbr.to_numpy(), hospital.patient_nbr.to_numpy()):
        raise ValueError("Reference patient IDs differ")
    if not np.array_equal(aligned.y_true.to_numpy(), hospital.target.to_numpy()):
        raise ValueError("Reference labels differ")
    return aligned.reset_index(drop=True)


def predict_chunks(model, x_test, chunk_rows):
    predictions = []
    for start in range(0, len(x_test), chunk_rows):
        stop = min(start + chunk_rows, len(x_test))
        predictions.append(model.predict_proba(x_test.iloc[start:stop])[:, 1])
        print(f"    predicted {stop:,}/{len(x_test):,}", flush=True)
    return np.concatenate(predictions)


def aligned_catboost(path, hospital, reference):
    source = pd.read_csv(path, usecols=["encounter_id", "patient_nbr", "Fold", "y_true", "catboost_raw"])
    if source.encounter_id.duplicated().any() or len(source) != len(hospital):
        raise ValueError("CatBoost OOF has a different cohort")
    aligned = source.set_index("encounter_id").reindex(hospital.encounter_id.to_numpy())
    if aligned.catboost_raw.isna().any():
        raise ValueError("CatBoost OOF is missing encounters")
    for name, expected in (("patient_nbr", hospital.patient_nbr),
                           ("Fold", reference.Fold), ("y_true", hospital.target)):
        if not np.array_equal(aligned[name].to_numpy(), expected.to_numpy()):
            raise ValueError(f"CatBoost OOF {name} disagrees with the reference")
    return aligned.catboost_raw.to_numpy(dtype=float)


def summarize(oof, models, output, alert_fraction):
    rows = []
    for fold, group in oof.groupby("Fold", sort=True):
        for name in models:
            rows.append({"Fold": int(fold), "Experiment": name,
                         "N_encounters": len(group), "N_patients": int(group.patient_nbr.nunique()),
                         **top_fraction_metrics(group.y_true, group[f"p_{name}"], alert_fraction)})
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("Experiment", sort=False).agg(
        Folds=("Fold", "size"), PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"), AUROC_mean=("AUROC", "mean"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        F1_mean=("F1_score", "mean"), Brier_mean=("Brier_Score", "mean"),
    ).reset_index()
    summary["Pooled_PR_AUC"] = [average_precision_score(oof.y_true, oof[f"p_{name}"])
                                 for name in summary.Experiment]
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def run_settings(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in {"resume", "audit_only"}}


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    base_columns = raw_discharge_predictors(hospital)
    ccs_map, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, ccs_map)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, n_events, n_linked = make_event_group_rates(
        events, event_ccs, args.min_group, args.alpha,
    )
    hospital = attach_group_features(hospital, rates, global_rates, EVENT_STATS)
    rates.to_csv(output / "meps_event_ccs_rates.csv", index=False)
    reference = aligned_reference(args.reference_oof, hospital)
    catboost_raw = aligned_catboost(args.catboost_oof, hospital, reference)
    base_columns += [f"ccs_{diag}" for diag in DIAG]
    print(f"[MEPS] events={n_events:,} linked_CCS={n_linked:,} groups={len(rates)}", flush=True)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(hospital, hospital.target, hospital.patient_nbr))
    audit = {"rows": len(hospital), "patients": int(hospital.patient_nbr.nunique()),
             "seed": args.seed, "folds": []}
    for fold, (train_ids, test_ids) in enumerate(splits, 1):
        if set(hospital.patient_nbr.iloc[train_ids]) & set(hospital.patient_nbr.iloc[test_ids]):
            raise ValueError(f"Patient overlap in fold {fold}")
        if not np.all(reference.Fold.to_numpy()[test_ids] == fold):
            raise ValueError(f"Reference fold assignment differs in fold {fold}")
        audit["folds"].append({"Fold": fold, "Train_Rows": len(train_ids),
                               "Test_Rows": len(test_ids),
                               "Test_Patients": int(hospital.patient_nbr.iloc[test_ids].nunique())})
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("[Audit] exact reference five-fold assignment and patient disjointness verified", flush=True)
    if args.audit_only:
        return
    TabPFNClassifier, selected_version, installed = get_tabpfn(args.model_version)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device == "cpu" and args.context_rows > 5000:
        raise ValueError("TabPFN 3.5 CPU context is limited to 5,000 rows; reduce --context-rows")
    print(f"[TabPFN] package={installed} model={args.model_version} device={device} "
          f"context_rows={args.context_rows} estimators={args.n_estimators}", flush=True)
    settings_path = output / "run_settings.json"
    settings = run_settings(args)
    if args.resume:
        if list(output.glob("oof_fold_*.csv")) and not settings_path.exists():
            raise ValueError("Saved fold outputs have no run settings; use a new --outdir")
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError("Saved TabPFN run settings differ; use a new --outdir")
    elif list(output.glob("oof_fold_*.csv")):
        raise ValueError("Saved fold outputs exist; pass --resume or use a new --outdir")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    parts = []
    for fold, (train_ids, test_ids) in enumerate(splits, 1):
        if fold > args.max_folds:
            break
        train, test = hospital.iloc[train_ids], hospital.iloc[test_ids]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient overlap in outer fold")
        if not np.all(reference.Fold.to_numpy()[test_ids] == fold):
            raise ValueError("Reference fold assignment differs")
        context_n = min(args.context_rows, len(train))
        if context_n == len(train):
            context_ids = np.arange(len(train))
        else:
            context_ids, _ = train_test_split(
                np.arange(len(train)), train_size=context_n,
                stratify=train.target.to_numpy(), random_state=args.seed + fold,
            )
        context = train.iloc[context_ids]
        if args.test_limit:
            # A pilot scores only a deterministic subset of the original held-out fold.
            subset = np.random.default_rng(args.seed + fold).choice(
                len(test), size=min(args.test_limit, len(test)), replace=False,
            )
            test = test.iloc[np.sort(subset)]
            test_ids = test_ids[np.sort(subset)]
        fold_path = output / f"oof_fold_{fold}.csv"
        if args.resume and fold_path.exists():
            saved = pd.read_csv(fold_path)
            needed = {"encounter_id", "patient_nbr", "y_true", "Fold",
                      "p_tabpfn35_ccs_only", "p_tabpfn35_ccs_meps"}
            if needed - set(saved) or len(saved) != len(test) or not np.array_equal(
                saved.encounter_id.to_numpy(), test.encounter_id.to_numpy()
            ) or not np.array_equal(saved.patient_nbr.to_numpy(), test.patient_nbr.to_numpy()) or not np.array_equal(
                saved.y_true.to_numpy(), test.target.to_numpy()
            ) or not saved.Fold.eq(fold).all():
                raise ValueError(f"Saved fold {fold} does not match the audited split")
            parts.append(saved)
            print(f"[Fold {fold}] resumed {len(saved):,} audited predictions", flush=True)
            continue
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        part["p_hgb_tuned"] = reference.iloc[test_ids].p_hgb_tuned.to_numpy()
        part["p_ccs_only_hgb"] = reference.iloc[test_ids].p_ccs_only_hgb.to_numpy()
        part["p_catboost_raw"] = catboost_raw[test_ids]
        part["p_ccs_only_hgb_catboost50"] = (part.p_ccs_only_hgb + part.p_catboost_raw) / 2
        for name, extra in VARIANTS.items():
            columns = base_columns + extra
            x_hgb_train, _, x_hgb_test = encode_frames(context, context, test, columns, 20)
            hgb = hgb_model(args.seed + fold, CONTEXT_HGB_PARAMS, early_stopping=False)
            hgb.fit(x_hgb_train, context.target.to_numpy())
            hgb_name = name.replace("tabpfn35", "context_hgb")
            part[f"p_{hgb_name}"] = hgb.predict_proba(x_hgb_test)[:, 1]
            x_context, categorical = tabpfn_frame(context, columns)
            x_test, _ = tabpfn_frame(test, columns)
            classifier = TabPFNClassifier.create_default_for_version(
                selected_version,
                device=device, n_estimators=args.n_estimators,
                categorical_features_indices=categorical,
                random_state=args.seed + fold,
                fit_mode="fit_with_cache",
            )
            print(f"[Fold {fold}] fitting {name}: context={len(context):,} test={len(test):,}", flush=True)
            classifier.fit(x_context, context.target.to_numpy())
            probability = predict_chunks(classifier, x_test, args.predict_chunk_rows)
            part[f"p_{name}"] = probability
            part[f"p_{name}_catboost50"] = (probability + part.p_catboost_raw.to_numpy()) / 2
            print(f"[Fold {fold}] {name} PR_AUC={average_precision_score(test.target, probability):.6f}", flush=True)
            del classifier
        parts.append(part)
        part.to_csv(fold_path, index=False)
        pd.concat(parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)

    oof = pd.concat(parts, ignore_index=True)
    full = args.max_folds == args.folds and args.test_limit == 0
    models = ["hgb_tuned", "ccs_only_hgb", "catboost_raw", "ccs_only_hgb_catboost50",
              "context_hgb_ccs_only", "context_hgb_ccs_meps", *VARIANTS,
              "tabpfn35_ccs_only_catboost50", "tabpfn35_ccs_meps_catboost50"]
    oof.to_csv(output / "oof_all.csv", index=False)
    summary = summarize(oof, models, output, args.alert_fraction)
    if full and args.bootstrap:
        bootstrap = cluster_bootstrap(oof, models, [
            ("tabpfn35_ccs_meps", "tabpfn35_ccs_only"),
            ("tabpfn35_ccs_only", "context_hgb_ccs_only"),
            ("tabpfn35_ccs_meps", "context_hgb_ccs_meps"),
            ("tabpfn35_ccs_meps", "ccs_only_hgb"),
            ("tabpfn35_ccs_meps_catboost50", "ccs_only_hgb_catboost50"),
            ("tabpfn35_ccs_meps_catboost50", "tabpfn35_ccs_only_catboost50"),
        ], args.bootstrap, args.seed + 17000)
        bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
        print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))
    elif not full:
        print("[Pilot] Partial folds/rows: these scores are not the full 5-fold result.", flush=True)
    (output / "config.json").write_text(json.dumps({**vars(args), "tabpfn_version": installed,
        "device_used": device, "full_evaluation": full,
        "threshold_policy": f"top {args.alert_fraction:.3f} fraction per outer test fold"},
        default=str, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--ccs-file", default=CCS_RAW)
    parser.add_argument("--event-data", default="data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", default="data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--reference-oof", default=str(ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv"))
    parser.add_argument("--catboost-oof", default=str(ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz"))
    parser.add_argument("--outdir", default=result_dir("tabpfn35_meps"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--test-limit", type=int, default=0, help="Rows per test fold; 0 means all")
    parser.add_argument("--context-rows", type=int, default=5000)
    parser.add_argument("--predict-chunk-rows", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--model-version", choices=["3.5", "3.5-fast"], default="3.5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--audit-only", action="store_true", help="Check exact folds and inputs without loading TabPFN")
    parser.add_argument("--resume", action="store_true", help="Reuse validated per-fold predictions after an interrupted run")
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if not (args.folds == 5 and 1 <= args.max_folds <= args.folds and args.test_limit >= 0
            and args.context_rows >= 100 and args.predict_chunk_rows >= 1 and args.n_estimators >= 1
            and args.min_group >= 1 and args.alpha >= 0 and args.bootstrap >= 0
            and 0 < args.alert_fraction < 1):
        parser.error("Invalid fold, context, prediction, or MEPS settings")
    for path in (args.hospital, args.ccs_file, args.event_data, args.event_ccs,
                 args.reference_oof, args.catboost_oof):
        if not Path(path).is_file():
            parser.error(f"Required file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
