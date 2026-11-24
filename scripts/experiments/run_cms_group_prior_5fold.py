#!/usr/bin/env python3
"""Evaluate real CMS national readmission aggregates on the fixed UCI five folds.

CMS rates are contextual features for four approximately matched principal CCS
diagnoses. They are neither patient outcomes nor calibrated UCI probabilities.
The original outer-fold HGB and CatBoost hyperparameters are held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import (  # noqa: E402
    catboost_shap_sample, load_comparison_cohort, make_paper_catboost_features,
)
from run_dual_external_meps_nhanes import (  # noqa: E402
    align_catboost, fit_fold, hospital_frame, load_reference, selected_hgb,
)
from run_paper_fair_comparison import (  # noqa: E402
    fit_catboost, paired_patient_bootstrap, save_shap, summarize,
)


KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
BASELINES = ["ccs_only_hgb", "catboost_raw", "ccs_only_hgb_catboost_raw_50"]
NEW_MODELS = ["cms_hgb", "cms_catboost_raw", "cms_hgb_catboost_raw_50",
              "ccs_only_hgb_cms_catboost_50", "cms_hgb_cms_catboost_50"]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cms(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    table = pd.read_csv(path)
    required = {"ccs", "cms_measure_id", "cms_national_rate_pct"}
    if missing := required - set(table):
        raise ValueError(f"CMS prior missing columns: {sorted(missing)}")
    if len(table) != 4 or table.ccs.duplicated().any():
        raise ValueError("Expected exactly four unique, predefined CMS-to-CCS mappings")
    values = pd.to_numeric(table.cms_national_rate_pct, errors="raise")
    if values.isna().any() or not values.between(0, 100).all():
        raise ValueError("CMS national percentages must be finite and within 0..100")
    lookup = dict(zip(table.ccs.astype(str), values.div(100).astype(float)))
    return table, lookup


def selected_catboost(path: Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    required = {"Fold", "Variant", "Depth", "Trees", "Validation_PR_AUC", "Validation_Brier"}
    if missing := required - set(trials):
        raise ValueError(f"CatBoost trials missing columns: {sorted(missing)}")
    raw = trials.loc[trials.Variant.eq("raw")].copy()
    raw = raw.sort_values(["Fold", "Validation_PR_AUC", "Validation_Brier", "Depth", "Trees"],
                          ascending=[True, False, True, True, True]).drop_duplicates("Fold")
    if set(raw.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected original CatBoost selections for all five folds")
    return raw.set_index("Fold")


def build_frames(args, oof: pd.DataFrame):
    hospital, hgb_columns = hospital_frame(args, oof)
    cms_table, lookup = read_cms(args.cms_prior)
    hospital["cms_readmission_rate"] = hospital.ccs_diag_1.map(lookup).fillna(-1.0).astype(float)
    hospital["cms_mapped"] = hospital.ccs_diag_1.isin(lookup).astype(int)
    covered = int(hospital.cms_mapped.sum())
    declared = int(cms_table.uci_principal_diagnosis_rows.sum()) if "uci_principal_diagnosis_rows" in cms_table else covered
    if covered != declared:
        raise ValueError(f"CMS coverage changed since lookup creation: {covered} vs {declared}")

    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(oof) or cohort.encounter_id.duplicated().any():
        raise ValueError("Raw CatBoost cohort differs from audited reference")
    cohort = cohort.set_index("encounter_id").loc[oof.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), oof.patient_nbr.to_numpy()) or not np.array_equal(
        cohort.target.to_numpy(), oof.y_true.to_numpy()
    ):
        raise ValueError("CatBoost cohort patient or label mismatch")
    catboost, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    catboost["cms_readmission_rate"] = hospital.cms_readmission_rate.to_numpy()
    catboost["cms_mapped"] = hospital.cms_mapped.to_numpy()
    return hospital, hgb_columns + ["cms_readmission_rate", "cms_mapped"], catboost, categorical, cms_table, covered


def run(args) -> None:
    if not 1 <= args.max_folds <= 5 or not 0 < args.alert_fraction < 1:
        raise ValueError("Use 1..5 folds and an alert fraction strictly between zero and one")
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference_oof)
    baseline_catboost = align_catboost(reference, args.catboost_oof)
    original = pd.read_csv(args.catboost_oof, usecols=KEYS + ["ccs_only_hgb"])
    original = original.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(original[KEYS].to_numpy(), reference[KEYS].to_numpy()):
        raise ValueError("Saved HGB+CatBoost baseline does not share the reference split")
    if not np.allclose(reference.p_ccs_only_hgb.to_numpy(), original.ccs_only_hgb.to_numpy(), atol=1e-12):
        raise ValueError("CCS-only HGB predictions differ between reference and paper comparison")
    expected_blend = (reference.p_ccs_only_hgb.to_numpy() + baseline_catboost) / 2
    hospital, hgb_columns, catboost, categorical, cms_table, covered = build_frames(args, reference)
    hgb_choice = selected_hgb(args.selected_hgb)
    cat_choice = selected_catboost(args.catboost_trials)
    folds = sorted(reference.Fold.unique())[:args.max_folds]
    output = reference.loc[reference.Fold.isin(folds), KEYS + ["p_ccs_only_hgb"]].copy()
    output = output.rename(columns={"p_ccs_only_hgb": "ccs_only_hgb"}).reset_index(drop=True)
    rows = np.flatnonzero(reference.Fold.isin(folds).to_numpy())
    output["catboost_raw"] = baseline_catboost[rows]
    output[BASELINES[2]] = expected_blend[rows]
    shap_tables = []
    for fold in folds:
        test = reference.Fold.eq(fold).to_numpy()
        train = ~test
        test_rows = output.Fold.eq(fold).to_numpy()
        if args.audit_only:
            continue
        hgb_pred = fit_fold(hospital, hgb_columns, train, test, hgb_choice, int(fold), args)
        x_train = catboost.loc[train].reset_index(drop=True)
        x_test = catboost.loc[test].reset_index(drop=True)
        model, cat_pred = fit_catboost(
            x_train, reference.loc[train, "y_true"].to_numpy(dtype=int),
            x_test, categorical, cat_choice.loc[fold], args,
        )
        output.loc[test_rows, "cms_hgb"] = hgb_pred
        output.loc[test_rows, "cms_catboost_raw"] = cat_pred
        output.loc[test_rows, "cms_hgb_catboost_raw_50"] = (hgb_pred + baseline_catboost[test]) / 2
        output.loc[test_rows, "ccs_only_hgb_cms_catboost_50"] = (
            reference.loc[test, "p_ccs_only_hgb"].to_numpy() + cat_pred
        ) / 2
        output.loc[test_rows, "cms_hgb_cms_catboost_50"] = (hgb_pred + cat_pred) / 2
        if args.shap_sample:
            shap = catboost_shap_sample(model, x_test, categorical, args.shap_sample,
                                        args.seed + int(fold) * 100)
            shap.insert(0, "Fold", int(fold))
            shap.insert(0, "Model", "cms_catboost_raw")
            shap_tables.append(shap)
        output.loc[test_rows].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)
        y = output.loc[test_rows, "y_true"]
        print(f"[Fold {fold}] CMS_HGB_AP={average_precision_score(y, hgb_pred):.6f} "
              f"CMS_CatBoost_AP={average_precision_score(y, cat_pred):.6f} "
              f"baseline_blend_AP={average_precision_score(y, expected_blend[test]):.6f}", flush=True)

    audit = {
        "reference_rows": int(len(reference)), "reference_patients": int(reference.patient_nbr.nunique()),
        "completed_folds": [int(x) for x in folds] if not args.audit_only else [],
        "patient_disjoint": bool(reference.groupby("patient_nbr").Fold.nunique().max() == 1),
        "cms_mapped_rows": covered, "cms_mapped_fraction": covered / len(reference),
        "cms_ccs_categories": cms_table.ccs.tolist(),
        "cms_rates_are": "national risk-standardized aggregate percentages, not UCI patient outcomes",
        "cms_period": cms_table[["cms_start_date", "cms_end_date"]].drop_duplicates().to_dict("records"),
        "cms_prior_sha256": checksum(args.cms_prior),
        "reference_oof_sha256": checksum(args.reference_oof),
        "catboost_oof_sha256": checksum(args.catboost_oof),
        "hgb_params": "original CCS-only selected parameters per fold",
        "catboost_params": "original raw-ICD validation choices per fold",
        "alert_policy": f"top {args.alert_fraction:.1%} by prediction in each fold",
        "seed": args.seed,
    }
    (args.outdir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    if args.audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    if output[NEW_MODELS].isna().any().any():
        raise ValueError("At least one test prediction is missing")
    output.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    summarize(output, BASELINES + NEW_MODELS, args.outdir, args.alert_fraction)
    paired_patient_bootstrap(output, [
        ("cms_hgb", "ccs_only_hgb"),
        ("cms_catboost_raw", "catboost_raw"),
        ("cms_hgb_catboost_raw_50", BASELINES[2]),
        ("ccs_only_hgb_cms_catboost_50", BASELINES[2]),
        ("cms_hgb_cms_catboost_50", BASELINES[2]),
    ], args.outdir, args.bootstrap, args.seed)
    save_shap(shap_tables, args.outdir)
    print(f"Saved CMS five-fold comparison to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--cms-prior", type=Path, default=ROOT / "data/processed/cms/cms_readmission_ccs_prior.csv")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--selected-hgb", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/selected_hgb.csv")
    parser.add_argument("--catboost-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--catboost-trials", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/catboost_inner_trials.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/cms_group_prior_5fold_seed777")
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--alert-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
