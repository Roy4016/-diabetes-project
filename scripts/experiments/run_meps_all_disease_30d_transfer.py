#!/usr/bin/env python3
"""All-disease MEPS 30-day transfer against the saved best CCS HGB/CatBoost.

Uses the same MEPS CCS-group teacher and optional joint neural training as
run_meps_30d_joint_transfer.py. UCI five-fold assignments and baseline OOF
probabilities remain fixed. All external features derive only from MEPS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.meps_30d_transfer import (  # noqa: E402
    common_matrix_meps, common_matrix_uci, group_support, load_events,
)
from diabetes_project.paper_comparison import make_paper_catboost_features, load_comparison_cohort  # noqa: E402
from run_ccs_catboost_5fold import prepare_features  # noqa: E402
from run_ccs_ensemble_imbalance_5fold import (  # noqa: E402
    BASE_BLEND, BASE_CAT, BASE_HGB, aligned_baseline, selected_catboost_trials,
)
from run_dual_external_meps_nhanes import hospital_frame, load_reference  # noqa: E402
from run_meps_30d_joint_transfer import (  # noqa: E402
    encoded_uci_private, fit_joint, fit_meps_teacher,
)
from run_paper_fair_comparison import paired_patient_bootstrap, summarize  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
TEACHER = "all_disease_meps_teacher_catboost"
TEACHER_BLEND = "all_disease_meps_teacher_hgb50"
DIABETES_TEACHER = "diabetes_meps_teacher_catboost"
DIABETES_BLEND = "diabetes_meps_teacher_hgb50"


def fit_teacher_catboost(frame, categorical, target, train, test, selected, fold, args):
    from catboost import CatBoostClassifier, Pool

    choice = selected.loc[fold]
    depth = int(choice.Depth)
    model = CatBoostClassifier(
        iterations=int(choice.Trees), depth=depth, learning_rate=.035,
        l2_leaf_reg={4: 5., 6: 15.}.get(depth, 15.),
        loss_function="Logloss", random_seed=2026, thread_count=args.threads,
        task_type="CPU", allow_writing_files=False, verbose=False,
    )
    model.fit(Pool(frame.loc[train], target[train], cat_features=categorical))
    return model.predict_proba(Pool(frame.loc[test], cat_features=categorical))[:, 1]


def saved_diabetes_teacher(reference, path, raw_cat):
    columns = KEYS + ["meps_teacher_risk", "meps_ccs_log_support"]
    saved = pd.read_csv(path, usecols=columns)
    if len(saved) != len(reference) or saved.encounter_id.duplicated().any():
        raise ValueError("Diabetes-only MEPS teacher OOF has different encounter coverage")
    saved = saved.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not saved[KEYS].equals(reference[KEYS]):
        raise ValueError("Diabetes-only MEPS teacher cohort, labels or folds differ")
    if saved[["meps_teacher_risk", "meps_ccs_log_support"]].isna().any().any():
        raise ValueError("Saved diabetes-only MEPS teacher features are missing")
    result = raw_cat.copy()
    result["meps_30d_teacher_risk"] = saved.meps_teacher_risk.to_numpy()
    result["meps_ccs_log_support"] = saved.meps_ccs_log_support.to_numpy()
    return result


def run(args):
    if args.seed != 777 or not 1 <= args.max_folds <= 5:
        raise ValueError("Saved best baseline requires seed 777 and 1..5 original folds")
    reference = load_reference(args.reference_oof)
    baseline = aligned_baseline(reference, args.baseline_oof)
    hospital, _ = hospital_frame(args, reference)
    cat_frame, categorical = prepare_features(args, reference)["append"]
    selected_cat = selected_catboost_trials(args.catboost_trials)
    diabetes_cat = saved_diabetes_teacher(reference, args.diabetes_teacher_oof, cat_frame)
    events, linked, groups = load_events(args.event_data, args.event_ccs, args.top_groups)
    if events.empty or linked.empty:
        raise ValueError("No labeled all-disease MEPS events with linked CCS records")
    common_uci = common_matrix_uci(hospital, groups)
    common_meps, private_meps = common_matrix_meps(events, linked, groups)
    support = group_support(common_uci, common_meps, groups)
    source_target = events.readmission_30d.to_numpy(dtype=int)
    target = reference.y_true.to_numpy(dtype=int)
    folds = list(range(1, args.max_folds + 1))
    active = reference.Fold.isin(folds).to_numpy()
    output = reference.loc[active, KEYS].copy().reset_index(drop=True)
    for name in (BASE_HGB, BASE_CAT, BASE_BLEND):
        output[name] = baseline.loc[active, name].to_numpy()
    print(f"[Audit] UCI={len(reference):,} encounters; 5 patient-disjoint folds; "
          f"MEPS={len(events):,} all-disease labeled events, "
          f"{events.person_key.nunique():,} people, {int(source_target.sum()):,} positives; "
          f"linked CCS={len(linked):,}; groups={len(groups)}", flush=True)
    if args.audit_only:
        return

    risk, diagnostic = fit_meps_teacher(
        common_meps, source_target, events.person_key.to_numpy(), common_uci, args,
    )
    cat_external = cat_frame.copy()
    cat_external["meps_30d_teacher_risk"] = risk
    cat_external["meps_ccs_log_support"] = support
    print(f"[MEPS teacher] validation AP={diagnostic['MEPS_Validation_PR_AUC']:.6f} "
          f"AUROC={diagnostic['MEPS_Validation_AUROC']:.6f}", flush=True)

    raw_cohort = load_comparison_cohort(args.hospital)
    raw_cohort = raw_cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(raw_cohort.target.to_numpy(), target):
        raise ValueError("NN raw UCI cohort label mismatch")
    raw_nn, raw_categories = make_paper_catboost_features(raw_cohort, raw_icd=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    training = []
    for fold in folds:
        test = reference.Fold.eq(fold).to_numpy()
        train = ~test
        local_test = output.Fold.eq(fold).to_numpy()
        p_cat = fit_teacher_catboost(cat_external, categorical, target, train, test,
                                     selected_cat, fold, args)
        output.loc[local_test, TEACHER] = p_cat
        output.loc[local_test, TEACHER_BLEND] = (
            p_cat + baseline.loc[test, BASE_HGB].to_numpy()) / 2
        p_diabetes = fit_teacher_catboost(diabetes_cat, categorical, target, train, test,
                                          selected_cat, fold, args)
        output.loc[local_test, DIABETES_TEACHER] = p_diabetes
        output.loc[local_test, DIABETES_BLEND] = (
            p_diabetes + baseline.loc[test, BASE_HGB].to_numpy()) / 2
        print(f"[Fold {fold}] {TEACHER} AP={average_precision_score(target[test], p_cat):.6f} "
              f"blend AP={average_precision_score(target[test], output.loc[local_test, TEACHER_BLEND]):.6f}",
              flush=True)
        if args.include_joint:
            private_uci = encoded_uci_private(raw_nn, raw_categories, train)
            for external, name in ((False, "uci_only_nn"), (True, "all_disease_joint_meps_30d")):
                p_nn, log = fit_joint(common_uci, private_uci, target, common_meps,
                                      private_meps, source_target, train, test,
                                      args, fold, external)
                output.loc[local_test, name] = p_nn
                output.loc[local_test, f"{name}_catboost50"] = (
                    p_nn + baseline.loc[test, BASE_CAT].to_numpy()) / 2
                training.extend(log)
                print(f"[Fold {fold}] {name} AP={average_precision_score(target[test], p_nn):.6f}",
                      flush=True)
        output.loc[local_test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)

    models = [BASE_HGB, BASE_CAT, BASE_BLEND, DIABETES_TEACHER, DIABETES_BLEND,
              TEACHER, TEACHER_BLEND]
    comparisons = [(DIABETES_TEACHER, BASE_CAT), (DIABETES_BLEND, BASE_BLEND),
                   (TEACHER, BASE_CAT), (TEACHER_BLEND, BASE_BLEND),
                   (TEACHER, DIABETES_TEACHER), (TEACHER_BLEND, DIABETES_BLEND)]
    if args.include_joint:
        models += ["uci_only_nn", "all_disease_joint_meps_30d",
                   "uci_only_nn_catboost50", "all_disease_joint_meps_30d_catboost50"]
        comparisons += [
            ("all_disease_joint_meps_30d", "uci_only_nn"),
            ("all_disease_joint_meps_30d_catboost50", "uci_only_nn_catboost50"),
            ("all_disease_joint_meps_30d_catboost50", BASE_BLEND),
        ]
        pd.DataFrame(training).to_csv(args.outdir / "joint_training.csv", index=False)
    if output[models].isna().any().any():
        raise ValueError("Incomplete held-out predictions")
    output.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    summarize(output, models, args.outdir, args.alert_fraction)
    paired_patient_bootstrap(output, comparisons, args.outdir, args.bootstrap, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"completed_folds": folds, "meps_events": len(events),
                   "meps_people": int(events.person_key.nunique()),
                   "meps_positive": int(source_target.sum()),
                   "linked_ccs_records": len(linked), "ccs_groups": groups,
                   "teacher_validation": diagnostic,
                   "baseline": BASE_BLEND,
                   "diabetes_teacher_oof": str(args.diabetes_teacher_oof),
                   "diabetes_control": "same best CatBoost inputs and settings, saved diabetic MEPS teacher features",
                   "external_catboost_features": ["meps_30d_teacher_risk", "meps_ccs_log_support"],
                   "external_label": "all-disease MEPS exact-date 30-day inpatient readmission",
                   "baseline_hgb": "saved CCS-only HGB OOF",
                   "catboost_selection": "saved outer-fold train-only hyperparameters",
                   "fusion": "0.5 * saved HGB + 0.5 * augmented CatBoost",
                   "threshold": f"top {args.alert_fraction:.0%} per outer test fold"})
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved to {args.outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--baseline-oof", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--catboost-trials", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/catboost_inner_trials.csv")
    parser.add_argument("--diabetes-teacher-oof", type=Path, default=ROOT / "results/meps_30d_joint_transfer_seed777/oof_all.csv.gz")
    parser.add_argument("--event-data", type=Path, default=ROOT / "data/processed/meps_events_all_disease_2008_2012/meps_all_disease_event_30d_model.csv")
    parser.add_argument("--event-ccs", type=Path, default=ROOT / "data/processed/meps_events_all_disease_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/meps_all_disease_30d_transfer_seed777")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--top-groups", type=int, default=64)
    parser.add_argument("--teacher-iterations", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--include-joint", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--external-weight", type=float, default=.3)
    parser.add_argument("--meps-interval", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.top_groups < 1 or args.teacher_iterations < 1 or args.threads < 1 or args.bootstrap < 0:
        parser.error("Invalid hyperparameters")
    run(args)


if __name__ == "__main__":
    main()
