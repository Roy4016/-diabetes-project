#!/usr/bin/env python3
"""Compare saved five-fold HGB OOF with fold-local CatBoost and SHAP.

The split is exactly the 99,343-row seed-777 OOF split used by the latest
CCS/MEPS HGB experiment. The local SHAP extension supplies the analysis idea;
its 80/20 predictions are kept out of this five-fold comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diabetes_project.paper_comparison import (  # noqa: E402
    catboost_shap_sample, hgb_shap_sample, load_comparison_cohort, make_paper_catboost_features,
    top_fraction_metrics,
)

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
EXISTING = ["bhuvan_hgb", "roy_hgb_46", "ccs_only_hgb", "ccs_meps_group_hgb"]
UCI_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_audit(args):
    source_hash = sha256(args.input)
    if source_hash != UCI_SHA256:
        raise ValueError(f"UCI source SHA-256 differs from the paper reproduction: {source_hash}")
    own = pd.read_csv(args.our_oof)
    bhuvan = pd.read_csv(args.bhuvan_oof)
    expected_own = KEYS + ["p_ccs_only_hgb", "p_ccs_meps_group_hgb"]
    expected_bhuvan = KEYS + ["bhuvan_hgb", "roy_hgb_46"]
    for name, table, expected in (("our", own, expected_own), ("Bhuvan", bhuvan, expected_bhuvan)):
        if missing := set(expected) - set(table):
            raise ValueError(f"{name} OOF missing columns: {sorted(missing)}")
        if table.encounter_id.duplicated().any():
            raise ValueError(f"{name} OOF has duplicate encounters")
    comparison = own[expected_own].merge(bhuvan[expected_bhuvan], on="encounter_id",
                                          how="outer", validate="one_to_one", suffixes=("", "_bhu"), indicator=True)
    if not comparison._merge.eq("both").all() or len(comparison) != 99343:
        raise ValueError("Five-fold OOF files have different cohorts")
    for name in KEYS[1:]:
        if not comparison[name].eq(comparison[f"{name}_bhu"]).all():
            raise ValueError(f"Five-fold OOF mismatch in {name}")
    comparison = comparison.drop(columns=["_merge"] + [f"{name}_bhu" for name in KEYS[1:]])
    comparison = comparison.rename(columns={
        "p_ccs_only_hgb": "ccs_only_hgb",
        "p_ccs_meps_group_hgb": "ccs_meps_group_hgb",
    })
    cohort = load_comparison_cohort(args.input)
    source_keys = cohort[["encounter_id", "patient_nbr", "target"]]
    check = comparison[KEYS[:3]].merge(source_keys, on="encounter_id", how="left",
                                        validate="one_to_one", suffixes=("", "_source"))
    if check.target.isna().any() or not check.patient_nbr.eq(check.patient_nbr_source).all():
        raise ValueError("Source rows do not match OOF patients")
    if not check.y_true.eq(check.target).all():
        raise ValueError("Source labels do not match OOF labels")
    if set(comparison.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected exactly five outer folds")
    if comparison.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Patient leakage across outer folds")
    if not np.isfinite(comparison[EXISTING].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite saved OOF prediction")
    return comparison, cohort, {
        "cohort_rows": int(len(comparison)),
        "patients": int(comparison.patient_nbr.nunique()),
        "positive_rate": float(comparison.y_true.mean()),
        "fold_rows": {str(k): int(v) for k, v in comparison.Fold.value_counts().sort_index().items()},
        "split_match": "exact encounter, patient, label and fold equality across both OOF files",
        "source_sha256": source_hash,
        "our_oof_sha256": sha256(args.our_oof),
        "bhuvan_oof_sha256": sha256(args.bhuvan_oof),
        "salim_result_status": "80/20 holdout only; SHAP/metrics extension of Rajpoot, not a five-fold Salim model",
        "historical_catboost_status": "80/20 holdout only; 0.251383 AP cannot be ranked against five-fold mean AP",
    }


def choose_catboost(x_train, y_train, groups, categorical, fold, args):
    from catboost import CatBoostClassifier, Pool

    inner, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                                random_state=args.seed + fold).split(
        x_train, y_train, groups=groups))
    if set(groups[inner]) & set(groups[validation]):
        raise ValueError("Patient leakage in CatBoost validation split")
    train_pool = Pool(x_train.iloc[inner], y_train[inner], cat_features=categorical)
    val_pool = Pool(x_train.iloc[validation], cat_features=categorical)
    trials = []
    for depth in args.depths:
        model = CatBoostClassifier(
            iterations=max(args.iterations), depth=depth,
            learning_rate=0.035, l2_leaf_reg={4: 5.0, 6: 15.0}.get(depth, 15.0),
            loss_function="Logloss", random_seed=2026, thread_count=args.threads,
            task_type="CPU", allow_writing_files=False, verbose=False,
        )
        model.fit(train_pool)
        for trees in args.iterations:
            p = model.predict_proba(val_pool, ntree_end=trees)[:, 1]
            trials.append({"Fold": fold, "Depth": depth, "Trees": trees,
                           "Validation_PR_AUC": float(average_precision_score(y_train[validation], p)),
                           "Validation_Brier": float(brier_score_loss(y_train[validation], p)),
                           "Validation_Patients": int(len(set(groups[validation])))})
    chosen = sorted(trials, key=lambda row: (-row["Validation_PR_AUC"], row["Validation_Brier"],
                                             row["Depth"], row["Trees"]))[0]
    return chosen, trials


def fit_catboost(x_train, y_train, x_test, categorical, chosen, args):
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(
        iterations=int(chosen["Trees"]), depth=int(chosen["Depth"]),
        learning_rate=0.035, l2_leaf_reg={4: 5.0, 6: 15.0}.get(int(chosen["Depth"]), 15.0),
        loss_function="Logloss", random_seed=2026, thread_count=args.threads,
        task_type="CPU", allow_writing_files=False, verbose=False,
    )
    model.fit(Pool(x_train, y_train, cat_features=categorical))
    return model, model.predict_proba(Pool(x_test, cat_features=categorical))[:, 1]


def prepare_our_meps_hgb(args, oof):
    """Rebuild the exact external event-CCS features from this project's code."""
    from run_cdc_encoder_control import prepare_hospital
    from run_meps_hgb import raw_discharge_predictors
    from run_meps_ccs_group_hgb import (DIAG, EVENT_STATS, attach_group_features,
                                        attach_uci_ccs, make_event_group_rates)
    from train_xgboost_ahrq_ccs_prefix import load_ccs

    hospital, _ = prepare_hospital(args.input)
    base = raw_discharge_predictors(hospital)
    mapping, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, mapping)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, _, _ = make_event_group_rates(events, event_ccs, 30, 50.0)
    hospital = attach_group_features(hospital, rates, global_rates, EVENT_STATS)
    columns = base + [f"ccs_{name}" for name in DIAG] + list(EVENT_STATS)
    if hospital.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter in MEPS HGB feature frame")
    hospital = hospital.set_index("encounter_id").loc[oof.encounter_id].reset_index()
    if not np.array_equal(hospital.patient_nbr.to_numpy(), oof.patient_nbr.to_numpy()):
        raise ValueError("MEPS HGB frame does not match OOF patients")
    selected = pd.read_csv(args.our_selected_hgb)
    selected = selected.loc[selected.Experiment.eq("ccs_meps_group_hgb")]
    if set(selected.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Missing a selected HGB configuration")
    return hospital, columns, selected.set_index("Fold")


def explain_our_meps_hgb(fold, oof, hospital, columns, selected, args):
    from run_nested_hgb_blend import encode_frames, hgb_model

    test = oof.Fold.eq(fold).to_numpy()
    train = ~test
    tr, te = hospital.loc[train], hospital.loc[test]
    x_train, _, x_test, transformer = encode_frames(tr, tr, te, columns, 100,
                                                     return_transformer=True)
    choice = selected.loc[fold]
    params = json.loads(choice.Parameters)
    params["max_iter"] = int(choice.Iterations)
    model = hgb_model(args.seed + int(fold), params, early_stopping=False)
    model.fit(x_train, oof.loc[train, "y_true"])
    probability = model.predict_proba(x_test)[:, 1]
    expected = oof.loc[test, "ccs_meps_group_hgb"].to_numpy()
    discrepancy = float(np.max(np.abs(probability - expected)))
    if discrepancy > 1e-8:
        raise ValueError(f"Fold {fold} MEPS HGB refit disagrees with the saved OOF: {discrepancy}")
    shap_table = hgb_shap_sample(model, x_test, transformer, columns, args.shap_sample,
                                 args.seed + int(fold) * 100 + 17)
    shap_table.insert(0, "Fold", int(fold))
    shap_table.insert(0, "Model", "ccs_meps_refit_hgb")
    return probability, discrepancy, shap_table


def summarize(oof, methods, outdir, fraction):
    records = []
    for fold, part in oof.groupby("Fold", sort=True):
        for name in methods:
            records.append({"Fold": int(fold), "Model": name, "Test_Rows": len(part),
                            "Test_Patients": int(part.patient_nbr.nunique()),
                            "Threshold_Policy": f"top_{fraction:g}_fraction_per_fold",
                            **top_fraction_metrics(part.y_true, part[name], fraction)})
    folds = pd.DataFrame(records)
    folds.to_csv(outdir / "fold_metrics.csv", index=False)
    summary = folds.groupby("Model", sort=False).agg(
        Folds=("Fold", "size"), PR_AUC_mean=("PR_AUC", "mean"), PR_AUC_std=("PR_AUC", "std"),
        AUROC_mean=("AUROC", "mean"), Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"), F1_mean=("F1_score", "mean"),
        Brier_mean=("Brier_Score", "mean"),
    ).reset_index()
    summary["Pooled_PR_AUC"] = [average_precision_score(oof.y_true, oof[name]) for name in summary.Model]
    summary.to_csv(outdir / "summary.csv", index=False)
    print(summary.round(6).to_string(index=False), flush=True)
    return summary


def save_shap(tables, outdir):
    if not tables:
        return
    all_folds = pd.concat(tables, ignore_index=True)
    all_folds.to_csv(outdir / "shap_fold.csv", index=False)
    weighted = []
    for (model, feature), part in all_folds.groupby(["Model", "Feature"], sort=False):
        weight = part.Explained_Rows.to_numpy(dtype=float)
        weighted.append({"Model": model, "Feature": feature,
                         "MeanAbsSHAP_logodds": float(np.average(part.MeanAbsSHAP_logodds, weights=weight)),
                         "MeanSHAP_logodds": float(np.average(part.MeanSHAP_logodds, weights=weight)),
                         "Explained_Rows": int(weight.sum())})
    global_table = pd.DataFrame(weighted).sort_values(["Model", "MeanAbsSHAP_logodds"],
                                                       ascending=[True, False])
    global_table["Rank"] = global_table.groupby("Model").cumcount() + 1
    global_table.to_csv(outdir / "shap_global.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, part in global_table.groupby("Model"):
        top = part.head(20).iloc[::-1]
        fig, axis = plt.subplots(figsize=(8, 7))
        axis.barh(top.Feature, top.MeanAbsSHAP_logodds)
        axis.set_xlabel("Mean absolute SHAP (log-odds)")
        axis.set_title(f"{name}: strongest features")
        fig.tight_layout()
        fig.savefig(outdir / f"shap_{name}.png", dpi=160)
        plt.close(fig)
    print("\nTop SHAP features\n", global_table.groupby("Model").head(10).to_string(index=False), flush=True)


def paired_patient_bootstrap(oof, pairs, outdir, replicates, seed):
    """Paired patient resampling within each outer fold; estimand is mean fold AP."""
    if replicates == 0:
        return
    rng = np.random.default_rng(seed + 17000)
    folds = []
    for _, part in oof.groupby("Fold", sort=True):
        _, inverse = np.unique(part.patient_nbr.to_numpy(), return_inverse=True)
        folds.append((part, inverse, int(inverse.max()) + 1))
    rows = []
    for treatment, control in pairs:
        observed = np.mean([
            average_precision_score(part.y_true, part[treatment]) -
            average_precision_score(part.y_true, part[control])
            for part, _, _ in folds
        ])
        samples = []
        for _ in range(replicates):
            changes = []
            for part, inverse, patients in folds:
                counts = np.bincount(rng.integers(0, patients, patients), minlength=patients)
                weights = counts[inverse]
                y = part.y_true.to_numpy()
                if np.unique(y[weights > 0]).size < 2:
                    raise ValueError("Patient bootstrap omitted one class")
                changes.append(average_precision_score(y, part[treatment], sample_weight=weights) -
                               average_precision_score(y, part[control], sample_weight=weights))
            samples.append(np.mean(changes))
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append({"Treatment": treatment, "Control": control,
                     "MeanFold_Delta_PR_AUC": observed, "CI95_Lower": low, "CI95_Upper": high,
                     "FractionPositive": float(np.mean(np.asarray(samples) > 0)),
                     "BootstrapReplicates": replicates, "ResamplingUnit": "patient within fold"})
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "paired_ap_bootstrap.csv", index=False)
    print("\nPaired AP differences\n", result.round(6).to_string(index=False), flush=True)


def run(args):
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    oof, cohort, audit = load_and_audit(args)
    (outdir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print("Split audit:", json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    if args.audit_only:
        summarize(oof, EXISTING, outdir, args.alert_fraction)
        return
    if args.report_only:
        saved = pd.read_csv(outdir / "oof_all.csv.gz")
        if len(saved) != len(oof) or not saved[KEYS].equals(oof[KEYS]):
            raise ValueError("Saved predictions do not match the audited five-fold split")
        from catboost import __version__ as catboost_version
        finalize_report(saved, args, outdir, catboost_version)
        return
    from catboost import __version__ as catboost_version
    if not args.skip_our_hgb_shap:
        import shap  # noqa: F401 - check availability before the long CatBoost training
    import sklearn

    x_family, categorical = make_paper_catboost_features(cohort)
    by_id = pd.Series(np.arange(len(cohort)), index=cohort.encounter_id)
    x_family = x_family.iloc[by_id.loc[oof.encounter_id].to_numpy()].reset_index(drop=True)
    matrices = {"family": x_family}
    if "raw" in args.variants:
        x_raw, raw_categorical = make_paper_catboost_features(cohort, raw_icd=True)
        if raw_categorical != categorical:
            raise ValueError("Raw ICD CatBoost has different feature types")
        matrices["raw"] = x_raw.iloc[by_id.loc[oof.encounter_id].to_numpy()].reset_index(drop=True)
    selections, shap_tables = [], []
    if not args.skip_our_hgb_shap:
        hospital, our_columns, our_selected = prepare_our_meps_hgb(args, oof)
    discrepancies = {}
    all_folds = [int(fold) for fold in sorted(oof.Fold.unique())[:args.max_folds]]
    for fold in all_folds:
        test = oof.Fold.eq(fold).to_numpy()
        train = ~test
        y_train = oof.loc[train, "y_true"].to_numpy(dtype=int)
        groups = oof.loc[train, "patient_nbr"].to_numpy()
        for variant in args.variants:
            x = matrices[variant]
            x_train, x_test = x.loc[train].reset_index(drop=True), x.loc[test].reset_index(drop=True)
            chosen, trials = choose_catboost(x_train, y_train, groups, categorical, int(fold), args)
            model, scores = fit_catboost(x_train, y_train, x_test, categorical, chosen, args)
            name = f"catboost_{variant}"
            oof.loc[test, name] = scores
            for row in trials:
                row["Variant"] = variant
                selections.append(row)
            shap_table = catboost_shap_sample(model, x_test, categorical, args.shap_sample,
                                              args.seed + int(fold) * 100 + (variant == "raw"))
            shap_table.insert(0, "Fold", int(fold))
            shap_table.insert(0, "Model", name)
            shap_tables.append(shap_table)
            print(f"Fold {fold} {name}: depth={chosen['Depth']} trees={chosen['Trees']} "
                  f"AP={average_precision_score(oof.loc[test, 'y_true'], scores):.6f}", flush=True)
        if "family" in args.variants:
            oof.loc[test, "bhuvan_hgb_catboost_50"] = (oof.loc[test, "bhuvan_hgb"] +
                                                        oof.loc[test, "catboost_family"]) / 2
            oof.loc[test, "ccs_meps_catboost_50"] = (oof.loc[test, "ccs_meps_group_hgb"] +
                                                      oof.loc[test, "catboost_family"]) / 2
        if not args.skip_our_hgb_shap:
            probability, discrepancy, shap_table = explain_our_meps_hgb(
                int(fold), oof, hospital, our_columns, our_selected, args)
            oof.loc[test, "ccs_meps_refit_hgb"] = probability
            discrepancies[str(fold)] = discrepancy
            shap_tables.append(shap_table)
            print(f"Fold {fold} ccs_meps_refit_hgb: max difference vs saved OOF={discrepancy:.3g}",
                  flush=True)
        oof.loc[test].to_csv(outdir / f"oof_fold_{fold}.csv.gz", index=False)
    fitted = oof.loc[oof.Fold.isin(all_folds)].copy()
    pd.DataFrame(selections).to_csv(outdir / "catboost_inner_trials.csv", index=False)
    save_shap(shap_tables, outdir)
    finalize_report(fitted, args, outdir, catboost_version, discrepancies)


def finalize_report(fitted, args, outdir, catboost_version, discrepancies=None):
    """Rebuild metrics from saved predictions without retraining the models."""
    import sklearn

    discrepancies = discrepancies or {}
    if not discrepancies and "ccs_meps_refit_hgb" in fitted:
        discrepancies = {str(int(fold)): float(np.max(np.abs(part.ccs_meps_refit_hgb -
                                                            part.ccs_meps_group_hgb)))
                         for fold, part in fitted.groupby("Fold")}
    variants = [variant for variant in ("family", "raw") if f"catboost_{variant}" in fitted]
    if not variants:
        raise ValueError("No CatBoost predictions found in saved OOF")
    if "family" in variants:
        fitted["bhuvan_hgb_catboost_50"] = (fitted.bhuvan_hgb + fitted.catboost_family) / 2
        fitted["ccs_only_catboost_50"] = (fitted.ccs_only_hgb + fitted.catboost_family) / 2
        fitted["ccs_meps_catboost_50"] = (fitted.ccs_meps_group_hgb + fitted.catboost_family) / 2
    if "raw" in variants:
        fitted["ccs_only_catboost_raw_50"] = (fitted.ccs_only_hgb + fitted.catboost_raw) / 2
        fitted["ccs_meps_catboost_raw_50"] = (fitted.ccs_meps_group_hgb + fitted.catboost_raw) / 2
    methods = EXISTING + [f"catboost_{name}" for name in variants]
    if "family" in variants:
        methods += ["bhuvan_hgb_catboost_50", "ccs_only_catboost_50", "ccs_meps_catboost_50"]
    if "raw" in variants:
        methods += ["ccs_only_catboost_raw_50", "ccs_meps_catboost_raw_50"]
    if "ccs_meps_refit_hgb" in fitted:
        methods += ["ccs_meps_refit_hgb"]
    if fitted[methods].isna().any().any():
        raise ValueError("Incomplete prediction matrix")
    fitted.to_csv(outdir / "oof_all.csv.gz", index=False)
    summarize(fitted, methods, outdir, args.alert_fraction)
    pairs = [(f"catboost_{variant}", "bhuvan_hgb") for variant in variants]
    pairs += [(f"catboost_{variant}", "ccs_only_hgb") for variant in variants]
    if "family" in variants:
        pairs += [("bhuvan_hgb_catboost_50", "bhuvan_hgb"),
                  ("ccs_meps_catboost_50", "ccs_meps_group_hgb"),
                  ("ccs_meps_catboost_50", "ccs_only_catboost_50")]
    if "raw" in variants:
        pairs += [("ccs_meps_catboost_raw_50", "ccs_only_catboost_raw_50"),
                  ("ccs_meps_catboost_raw_50", "catboost_raw")]
    paired_patient_bootstrap(fitted, pairs, outdir, args.bootstrap, args.seed)
    all_folds = [int(fold) for fold in sorted(fitted.Fold.unique())]
    previous = outdir / "run.json"
    old = json.loads(previous.read_text()) if previous.exists() else {}
    bootstrap_file = outdir / "paired_ap_bootstrap.csv"
    bootstrap_replicates = args.bootstrap
    if args.report_only and args.bootstrap == 0 and bootstrap_file.exists():
        bootstrap_replicates = int(pd.read_csv(bootstrap_file).BootstrapReplicates.iloc[0])
    metadata = {"models": methods, "completed_folds": all_folds,
                "partial_run": len(all_folds) != 5,
                "catboost_version": catboost_version or old.get("catboost_version") or "see installed package",
                "sklearn_version": sklearn.__version__,
                "threshold_policy": f"top {args.alert_fraction:.0%} per outer fold, no test labels used",
                "catboost_selection": "within each outer train: patient-disjoint validation, max AP then min Brier",
                "shap_scale": "CatBoost exact TreeSHAP, raw log-odds; global mean absolute magnitude",
                "our_hgb_shap": "TreeSHAP on a fold-specific refit; one-hot contributions summed to source fields",
                "our_hgb_refit_max_probability_difference_by_fold": discrepancies or old.get(
                    "our_hgb_refit_max_probability_difference_by_fold", {}),
                "patient_bootstrap_replicates": bootstrap_replicates,
                "historical_0_237_comparison": "Different Rajpoot 80/20 holdout; report descriptively, not as matched SOTA win"}
    (outdir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--our-oof", type=Path,
                        default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--bhuvan-oof", type=Path, default=ROOT.parent /
                        "Bhuvan_Rajpoot_SSRN/5-fold-reproduce/results/five_fold/baseline/oof_predictions.csv.gz")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--event-data", type=Path, default=ROOT /
                        "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", type=Path, default=ROOT /
                        "data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--our-selected-hgb", type=Path, default=ROOT /
                        "results/meps_event_ccs_group_hgb_seed777/selected_hgb.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/paper_fair_comparison_seed777")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild metrics and run.json from saved OOF; no model training")
    parser.add_argument("--max-folds", type=int, default=5, help="Pilot with 1-4; full comparison uses 5")
    parser.add_argument("--variants", choices=["family", "raw"], nargs="+", default=["family", "raw"])
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--iterations", type=int, nargs="+", default=[250, 500, 750])
    parser.add_argument("--shap-sample", type=int, default=1000,
                        help="Test encounters explained per fold per variant; 0 means all")
    parser.add_argument("--skip-our-hgb-shap", action="store_true",
                        help="Skip the refit and SHAP analysis of the project's external HGB")
    parser.add_argument("--alert-fraction", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Paired patient bootstrap replicates; 0 skips")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5 or not 0 < args.alert_fraction < 1:
        parser.error("--max-folds must be 1-5 and --alert-fraction must be between 0 and 1")
    if (args.threads < 1 or args.shap_sample < 0 or args.bootstrap < 0 or
            min(args.depths) < 1 or min(args.iterations) < 1):
        parser.error("Invalid threads, SHAP sample, depth or iteration count")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants")
    if args.audit_only and args.report_only:
        parser.error("Choose either --audit-only or --report-only")
    run(args)


if __name__ == "__main__":
    main()
