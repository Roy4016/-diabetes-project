#!/usr/bin/env python3
"""Evaluate CCS-aware CatBoost and CCS-HGB + CCS-CatBoost on fixed UCI folds.

The append variant retains raw ICD-9 and adds three AHRQ CCS categories. The
replace variant replaces raw ICD-9 fields with those same CCS categories.
CatBoost hyperparameters are selected within each outer training fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import (  # noqa: E402
    load_comparison_cohort, make_paper_catboost_features,
)
from run_cms_meps_cdc_catboost import aligned_saved  # noqa: E402
from run_dual_external_meps_nhanes import hospital_frame, load_reference  # noqa: E402
from run_meps_ccs_group_hgb import DIAG  # noqa: E402
from run_paper_fair_comparison import (  # noqa: E402
    choose_catboost, fit_catboost, paired_patient_bootstrap, summarize,
)

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
CCS_COLUMNS = [f"ccs_{name}" for name in DIAG]
NAMES = {"append": "catboost_raw_plus_ccs", "replace": "catboost_ccs_replace_diag"}


def prepare_features(args, reference: pd.DataFrame):
    hospital, _ = hospital_frame(args, reference)
    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(reference) or cohort.encounter_id.duplicated().any():
        raise ValueError("Raw CatBoost cohort differs from saved reference")
    cohort = cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), reference.patient_nbr.to_numpy()):
        raise ValueError("Patient IDs differ from saved reference")
    if not np.array_equal(cohort.target.to_numpy(), reference.y_true.to_numpy()):
        raise ValueError("Labels differ from saved reference")
    raw, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    if len(raw.columns) != 46:
        raise ValueError("Expected the original 46 discharge-time CatBoost features")
    ccs = hospital[CCS_COLUMNS].astype(str).reset_index(drop=True)
    appended = pd.concat([raw.reset_index(drop=True), ccs], axis=1)
    replaced = pd.concat([raw.drop(columns=list(DIAG)).reset_index(drop=True), ccs], axis=1)
    category_columns = list(dict.fromkeys([*categorical, *CCS_COLUMNS]))
    return {
        "append": (appended, category_columns),
        "replace": (replaced, [name for name in category_columns if name in replaced]),
    }


def run(args) -> None:
    if args.seed != 777:
        raise ValueError("The saved reference folds and baselines require seed 777")
    if not 1 <= args.max_folds <= 5 or not 0 < args.alert_fraction < 1:
        raise ValueError("Use 1..5 folds and 0 < alert-fraction < 1")
    if args.bootstrap < 0 or args.threads < 1 or not args.depths or not args.iterations:
        raise ValueError("Invalid bootstrap, threads or CatBoost trial grid")

    reference = load_reference(args.reference_oof)
    baseline = aligned_saved(reference, args.paper_oof, ["catboost_raw", "ccs_only_hgb"])
    if not np.allclose(reference.p_ccs_only_hgb.to_numpy(),
                       baseline.ccs_only_hgb.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError("Saved CCS-HGB probabilities disagree across OOF files")
    frames = prepare_features(args, reference)
    folds = [int(fold) for fold in sorted(reference.Fold.unique())[:args.max_folds]]
    selected = reference.Fold.isin(folds).to_numpy()
    oof = reference.loc[selected, KEYS].copy().reset_index(drop=True)
    oof["catboost_raw"] = baseline.loc[selected, "catboost_raw"].to_numpy()
    oof["ccs_only_hgb"] = baseline.loc[selected, "ccs_only_hgb"].to_numpy()
    oof["ccs_hgb_catboost_raw_50"] = (oof.ccs_only_hgb + oof.catboost_raw) / 2
    print(f"[Audit] rows={len(reference):,} patients={reference.patient_nbr.nunique():,} "
          f"folds={folds} patient_disjoint=verified", flush=True)
    print("[Features] append: raw ICD-9 + CCS; replace: CCS instead of raw ICD-9", flush=True)
    if args.audit_only:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    selections = []
    for fold in folds:
        train = reference.Fold.ne(fold).to_numpy()
        test = reference.Fold.eq(fold).to_numpy()
        output_test = oof.Fold.eq(fold).to_numpy()
        y_train = reference.loc[train, "y_true"].to_numpy(dtype=int)
        groups = reference.loc[train, "patient_nbr"].to_numpy()
        for variant in args.variants:
            frame, categorical = frames[variant]
            x_train = frame.loc[train].reset_index(drop=True)
            x_test = frame.loc[test].reset_index(drop=True)
            choice, trials = choose_catboost(
                x_train, y_train, groups, categorical, int(fold), args,
            )
            for trial in trials:
                selections.append({"Model": NAMES[variant], **trial})
            _, probability = fit_catboost(x_train, y_train, x_test, categorical, choice, args)
            name = NAMES[variant]
            oof.loc[output_test, name] = probability
            oof.loc[output_test, f"ccs_hgb_{name}_50"] = (
                oof.loc[output_test, "ccs_only_hgb"].to_numpy() + probability
            ) / 2
            print(f"[Fold {fold}] {name} depth={choice['Depth']} trees={choice['Trees']}",
                  flush=True)
        oof.loc[output_test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)

    methods = ["catboost_raw", "ccs_only_hgb", "ccs_hgb_catboost_raw_50"]
    for variant in args.variants:
        methods.extend([NAMES[variant], f"ccs_hgb_{NAMES[variant]}_50"])
    if oof[methods].isna().any().any():
        raise ValueError("An outer-fold prediction is missing")
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    pd.DataFrame(selections).to_csv(args.outdir / "catboost_inner_trials.csv", index=False)
    summarize(oof, methods, args.outdir, args.alert_fraction)
    comparisons = []
    for variant in args.variants:
        name = NAMES[variant]
        comparisons.extend([
            (name, "catboost_raw"),
            (f"ccs_hgb_{name}_50", "ccs_hgb_catboost_raw_50"),
        ])
    if len(args.variants) == 2:
        comparisons.extend([
            (NAMES["append"], NAMES["replace"]),
            (f"ccs_hgb_{NAMES['append']}_50", f"ccs_hgb_{NAMES['replace']}_50"),
        ])
    paired_patient_bootstrap(oof, comparisons, args.outdir, args.bootstrap, args.seed)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "folds": folds, "reference_oof": str(args.reference_oof),
        "paper_oof": str(args.paper_oof), "hospital": str(args.hospital),
        "ccs_file": str(args.ccs_file), "variants": args.variants,
        "features": {variant: frames[variant][0].columns.tolist() for variant in args.variants},
        "selection": "per-fold patient-disjoint inner validation; maximize AP",
        "depths": args.depths, "iterations": args.iterations,
        "metric_policy": f"top {args.alert_fraction:.0%} per test fold for Precision/Recall/F1",
        "bootstrap": args.bootstrap,
    }, indent=2) + "\n")
    print(f"Saved CCS CatBoost comparison to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--paper-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777")
    parser.add_argument("--variants", choices=list(NAMES), nargs="+", default=list(NAMES))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--iterations", type=int, nargs="+", default=[250, 500, 750])
    parser.add_argument("--alert-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
