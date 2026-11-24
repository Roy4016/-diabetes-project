#!/usr/bin/env python3
"""Compare balanced weights and SMOTENC for the saved best CCS HGB/CatBoost blend.

Only outer-training encounters are weighted or resampled. The original foldwise
HGB/CatBoost hyperparameters and fixed 50/50 probability blend are retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import (  # noqa: E402
    load_comparison_cohort, make_paper_catboost_features,
)
from run_dual_external_meps_nhanes import hospital_frame, load_reference, selected_hgb  # noqa: E402
from run_meps_ccs_group_hgb import DIAG  # noqa: E402
from run_nested_hgb_blend import encode_frames, hgb_model  # noqa: E402
from run_paper_fair_comparison import paired_patient_bootstrap, summarize  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
BASE_HGB = "ccs_only_hgb"
BASE_CAT = "catboost_raw_plus_ccs"
BASE_BLEND = "ccs_hgb_catboost_raw_plus_ccs_50"
METHODS = ("class_weight", "smotenc")


def aligned_baseline(reference: pd.DataFrame, path: Path) -> pd.DataFrame:
    saved = pd.read_csv(path, usecols=KEYS + [BASE_HGB, BASE_CAT, BASE_BLEND])
    if len(saved) != len(reference) or saved.encounter_id.duplicated().any():
        raise ValueError("Best CCS ensemble OOF has different encounter coverage")
    saved = saved.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not saved[KEYS].equals(reference[KEYS]):
        raise ValueError("Best CCS ensemble has different patients, labels or folds")
    if saved[[BASE_HGB, BASE_CAT, BASE_BLEND]].isna().any().any():
        raise ValueError("Best CCS ensemble OOF has missing scores")
    if not np.allclose(saved[BASE_BLEND], (saved[BASE_HGB] + saved[BASE_CAT]) / 2,
                       atol=1e-12, rtol=0):
        raise ValueError("Saved CCS ensemble is not the fixed 50/50 blend")
    if not np.allclose(saved[BASE_HGB], reference.p_ccs_only_hgb, atol=1e-12, rtol=0):
        raise ValueError("Saved CCS HGB differs from reference OOF")
    return saved


def frames(args, reference: pd.DataFrame):
    hospital, hgb_columns = hospital_frame(args, reference)
    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(reference) or cohort.encounter_id.duplicated().any():
        raise ValueError("CatBoost cohort differs from reference")
    cohort = cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), reference.patient_nbr.to_numpy()) \
            or not np.array_equal(cohort.target.to_numpy(), reference.y_true.to_numpy()):
        raise ValueError("CatBoost cohort patient or label mismatch")
    cat, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    ccs_columns = [f"ccs_{name}" for name in DIAG]
    for name in ccs_columns:
        cat[name] = hospital[name].astype(str).to_numpy()
    categorical = [*categorical, *ccs_columns]
    if len(hgb_columns) != 49 or len(cat.columns) != 49:
        raise ValueError("Expected 46 discharge predictors and 3 CCS categories per model")
    return hospital, hgb_columns, cat, categorical


def smotenc_resample(frame: pd.DataFrame, labels: np.ndarray,
                     categorical: list[str], ratio: float, seed: int):
    """Fit imputers/scaler on this outer-training partition, then synthesize there."""
    from imblearn.over_sampling import SMOTENC

    labels = np.asarray(labels, dtype=int)
    current_ratio = float(np.sum(labels == 1) / np.sum(labels == 0))
    if ratio <= current_ratio:
        raise ValueError(f"SMOTENC target ratio {ratio:g} must exceed training ratio {current_ratio:.4f}")
    category_set = set(categorical)
    numeric = [name for name in frame if name not in category_set]
    clean = frame.copy().reset_index(drop=True)
    if numeric:
        numeric_imputer = SimpleImputer(strategy="median")
        numeric_values = numeric_imputer.fit_transform(clean[numeric].apply(pd.to_numeric, errors="coerce"))
        scaler = StandardScaler()
        clean[numeric] = scaler.fit_transform(numeric_values)
    else:
        scaler = None
    for name in categorical:
        clean[name] = clean[name].astype(object).where(clean[name].notna(), "__MISSING__").astype(str)
    category_ids = [clean.columns.get_loc(name) for name in categorical]
    sampler = SMOTENC(categorical_features=category_ids, sampling_strategy=ratio,
                      k_neighbors=5, random_state=seed)
    resampled, target = sampler.fit_resample(clean, labels)
    resampled = pd.DataFrame(resampled, columns=clean.columns)
    if numeric:
        resampled[numeric] = scaler.inverse_transform(resampled[numeric].astype(float))
    for name in categorical:
        resampled[name] = resampled[name].astype(str)
    if not np.isfinite(resampled[numeric].to_numpy(dtype=float)).all():
        raise ValueError("SMOTENC generated nonfinite numeric features")
    return resampled, np.asarray(target, dtype=int)


def hgb_categorical(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    category_ids = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    return [name for name in columns if name in category_ids
            or pd.api.types.is_object_dtype(frame[name])
            or pd.api.types.is_string_dtype(frame[name])]


def fit_hgb(train_frame, test_frame, columns, y_train, fold, selection, args,
            method, ratio):
    if method == "smotenc":
        category_columns = hgb_categorical(train_frame, columns)
        train_frame, y_train = smotenc_resample(
            train_frame[columns], y_train, category_columns,
            args.smote_ratio, args.seed + fold * 100 + 1,
        )
        test_frame = test_frame[columns].copy()
        for name in category_columns:
            test_frame[name] = (test_frame[name].astype(object)
                                .where(test_frame[name].notna(), "__MISSING__").astype(str))
    x_train, _, x_test = encode_frames(train_frame, train_frame, test_frame,
                                      columns, args.min_frequency)
    chosen = selection.loc[fold]
    params = json.loads(chosen.Parameters)
    params["max_iter"] = int(chosen.Iterations)
    model = hgb_model(args.seed + fold, params, early_stopping=False)
    if method == "class_weight":
        weights = np.where(y_train == 1, ratio, 1.0)
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1], len(y_train)


def selected_catboost_trials(path: Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    needed = {"Model", "Fold", "Depth", "Trees", "Validation_PR_AUC", "Validation_Brier"}
    if needed - set(trials):
        raise ValueError(f"Missing CatBoost trial columns: {sorted(needed - set(trials))}")
    selected = trials[trials.Model.eq(BASE_CAT)].sort_values(
        ["Fold", "Validation_PR_AUC", "Validation_Brier", "Depth", "Trees"],
        ascending=[True, False, True, True, True],
    ).drop_duplicates("Fold").set_index("Fold")
    if set(selected.index) != {1, 2, 3, 4, 5}:
        raise ValueError("Need saved CatBoost+CCS inner selections in all five folds")
    return selected


def fit_catboost(train_frame, test_frame, categorical, y_train, fold, selection,
                 args, method, ratio):
    from catboost import CatBoostClassifier, Pool

    if method == "smotenc":
        train_frame, y_train = smotenc_resample(
            train_frame, y_train, categorical, args.smote_ratio,
            args.seed + fold * 100 + 2,
        )
    selected = selection.loc[fold]
    depth = int(selected.Depth)
    options = dict(
        iterations=int(selected.Trees), depth=depth, learning_rate=0.035,
        l2_leaf_reg={4: 5.0, 6: 15.0}.get(depth, 15.0),
        loss_function="Logloss", random_seed=2026, thread_count=args.threads,
        task_type="CPU", allow_writing_files=False, verbose=False,
    )
    if method == "class_weight":
        options["class_weights"] = [1.0, ratio]
    model = CatBoostClassifier(**options)
    model.fit(Pool(train_frame, y_train, cat_features=categorical))
    probability = model.predict_proba(Pool(test_frame, cat_features=categorical))[:, 1]
    return probability, len(y_train)


def run(args):
    if args.seed != 777 or not 1 <= args.max_folds <= 5:
        raise ValueError("Use seed 777 and 1..5 original folds")
    if args.smote_ratio <= 0 or args.bootstrap < 0 or args.threads < 1:
        raise ValueError("Invalid SMOTENC ratio, bootstrap count or thread count")
    reference = load_reference(args.reference_oof)
    baseline = aligned_baseline(reference, args.baseline_oof)
    hospital, hgb_columns, cat, categorical = frames(args, reference)
    hgb_selection = selected_hgb(args.hgb_selection)
    cat_selection = selected_catboost_trials(args.catboost_trials)
    folds = [int(x) for x in sorted(reference.Fold.unique())[:args.max_folds]]
    selected = reference.Fold.isin(folds).to_numpy()
    oof = reference.loc[selected, KEYS].copy().reset_index(drop=True)
    for name in (BASE_HGB, BASE_CAT, BASE_BLEND):
        oof[name] = baseline.loc[selected, name].to_numpy()
    print(f"[Audit] rows={len(reference):,} patients={reference.patient_nbr.nunique():,} "
          f"folds={folds}; baseline={BASE_BLEND}; fixed HGB/CatBoost selections", flush=True)
    if args.audit_only:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    training_rows = []
    for fold in folds:
        train = reference.Fold.ne(fold).to_numpy()
        test = reference.Fold.eq(fold).to_numpy()
        output_test = oof.Fold.eq(fold).to_numpy()
        y_train = reference.loc[train, "y_true"].to_numpy(dtype=int)
        ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
        for method in args.methods:
            hgb_p, hgb_n = fit_hgb(hospital.loc[train], hospital.loc[test],
                                   hgb_columns, y_train, fold, hgb_selection,
                                   args, method, ratio)
            cat_p, cat_n = fit_catboost(cat.loc[train].reset_index(drop=True),
                                        cat.loc[test].reset_index(drop=True),
                                        categorical, y_train, fold, cat_selection,
                                        args, method, ratio)
            hgb_name, cat_name, blend_name = (f"ccs_hgb_{method}",
                                               f"ccs_catboost_{method}",
                                               f"ccs_hgb_catboost_{method}_50")
            oof.loc[output_test, hgb_name] = hgb_p
            oof.loc[output_test, cat_name] = cat_p
            oof.loc[output_test, blend_name] = (hgb_p + cat_p) / 2
            training_rows.append({"Fold": fold, "Method": method,
                                  "Train_Rows": int(train.sum()),
                                  "Train_Positive_Rate": float(y_train.mean()),
                                  "Positive_Weight": ratio if method == "class_weight" else 1.0,
                                  "SMOTENC_Target_Positive_Negative_Ratio":
                                  args.smote_ratio if method == "smotenc" else None,
                                  "HGB_Fit_Rows": hgb_n, "CatBoost_Fit_Rows": cat_n})
            print(f"[Fold {fold}] {method}: HGB rows={hgb_n:,}, "
                  f"CatBoost rows={cat_n:,}, positive weight={ratio:.3f}", flush=True)
        oof.loc[output_test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)

    models = [BASE_HGB, BASE_CAT, BASE_BLEND]
    for method in args.methods:
        models += [f"ccs_hgb_{method}", f"ccs_catboost_{method}",
                   f"ccs_hgb_catboost_{method}_50"]
    if oof[models].isna().any().any():
        raise ValueError("A test-fold prediction is missing")
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    pd.DataFrame(training_rows).to_csv(args.outdir / "training_audit.csv", index=False)
    summarize(oof, models, args.outdir, args.alert_fraction)
    comparisons = []
    for method in args.methods:
        comparisons += [(f"ccs_hgb_{method}", BASE_HGB),
                        (f"ccs_catboost_{method}", BASE_CAT),
                        (f"ccs_hgb_catboost_{method}_50", BASE_BLEND)]
    if set(args.methods) == set(METHODS):
        comparisons.append(("ccs_hgb_catboost_smotenc_50", "ccs_hgb_catboost_class_weight_50"))
    paired_patient_bootstrap(oof, comparisons, args.outdir, args.bootstrap, args.seed)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "folds": folds, "reference_oof": str(args.reference_oof),
        "baseline_oof": str(args.baseline_oof), "hgb_selection": str(args.hgb_selection),
        "catboost_trials": str(args.catboost_trials), "hospital": str(args.hospital),
        "ccs_file": str(args.ccs_file), "baseline": BASE_BLEND,
        "methods": args.methods, "smotenc_target_ratio": args.smote_ratio,
        "class_weight": "train-fold negative/positive count; applied to both learners",
        "hgb_features": hgb_columns, "catboost_features": cat.columns.tolist(),
        "hyperparameters": "fixed saved per-fold HGB/CatBoost selections",
        "alert_fraction": args.alert_fraction, "bootstrap": args.bootstrap,
    }, indent=2) + "\n")
    print(f"Saved results to {args.outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--baseline-oof", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--hgb-selection", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/selected_hgb.csv")
    parser.add_argument("--catboost-trials", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/catboost_inner_trials.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/ccs_ensemble_imbalance_5fold_seed777")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--smote-ratio", type=float, default=0.20)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--alert-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
