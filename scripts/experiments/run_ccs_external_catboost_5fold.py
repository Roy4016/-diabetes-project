#!/usr/bin/env python3
"""Compare CDC/CMS augmentation of raw and CCS CatBoost with saved CCS HGB.

The raw CatBoost variants and HGB are read from audited five-fold OOF files.
When requested, a CatBoost CDC teacher replaces the historical XGBoost proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import load_comparison_cohort, make_paper_catboost_features  # noqa: E402
from run_cdc_xgb import BRIDGE_COLS, load_cdc  # noqa: E402
from run_cms_group_prior_5fold import read_cms, selected_catboost  # noqa: E402
from run_cms_meps_cdc_catboost import aligned_saved, cdc_proxy  # noqa: E402
from run_dual_external_meps_nhanes import hospital_frame, load_reference  # noqa: E402
from run_meps_ccs_group_hgb import DIAG  # noqa: E402
from run_paper_fair_comparison import fit_catboost, paired_patient_bootstrap, summarize  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
SOURCES = ("cdc", "cms")


def cdc_catboost_proxy(args, hospital: pd.DataFrame) -> tuple[np.ndarray, dict]:
    from catboost import CatBoostClassifier

    source = load_cdc(args.cdc)
    source = source.loc[source.Diabetes_012.eq(2)].copy()
    if len(source) < 1000 or source.poor_health.nunique() != 2:
        raise ValueError("Too few labelled diabetic CDC respondents")
    if hospital[BRIDGE_COLS].isna().all(axis=1).any():
        raise ValueError("A UCI encounter has no CDC bridge fields")
    x = source[BRIDGE_COLS].astype(float)
    y = source.poor_health.astype(int)
    train, validation = train_test_split(
        np.arange(len(source)), test_size=.20, stratify=y, random_state=args.seed,
    )

    def model() -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=300, depth=3, learning_rate=.05, l2_leaf_reg=2.,
            loss_function="Logloss", random_seed=args.seed,
            thread_count=args.threads, task_type="CPU",
            allow_writing_files=False, verbose=False,
        )

    check = model().fit(x.iloc[train], y.iloc[train])
    val_score = check.predict_proba(x.iloc[validation])[:, 1]
    final = model().fit(x, y)
    score = final.predict_proba(hospital[BRIDGE_COLS].astype(float))[:, 1]
    if not np.isfinite(score).all():
        raise ValueError("CDC CatBoost teacher produced nonfinite scores")
    audit = {
        "teacher": "CatBoost", "respondents": int(len(source)),
        "target": "CDC self-rated general health fair/poor; not readmission",
        "bridge_features": list(BRIDGE_COLS),
        "validation_rows": int(len(validation)),
        "held_out_cdc_ap": float(average_precision_score(y.iloc[validation], val_score)),
        "held_out_cdc_auroc": float(roc_auc_score(y.iloc[validation], val_score)),
        "parameters": {"iterations": 300, "depth": 3, "learning_rate": .05, "l2_leaf_reg": 2.},
    }
    print(f"[CDC CatBoost teacher] respondents={len(source):,} "
          f"validation_AP={audit['held_out_cdc_ap']:.6f}", flush=True)
    return score, audit


def chosen_ccs_parameters(path: Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    required = {"Model", "Fold", "Depth", "Trees", "Validation_PR_AUC", "Validation_Brier"}
    if missing := required - set(trials):
        raise ValueError(f"CCS CatBoost selections are missing {sorted(missing)}")
    trials = trials.loc[trials.Model.eq("catboost_raw_plus_ccs")]
    choice = trials.sort_values(
        ["Fold", "Validation_PR_AUC", "Validation_Brier", "Depth", "Trees"],
        ascending=[True, False, True, True, True],
    ).drop_duplicates("Fold").set_index("Fold")
    if set(choice.index) != {1, 2, 3, 4, 5}:
        raise ValueError("Need saved CCS CatBoost inner selections for all five folds")
    return choice


def make_features(args, reference: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    hospital, _ = hospital_frame(args, reference)
    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(reference) or cohort.encounter_id.duplicated().any():
        raise ValueError("Raw CatBoost cohort differs from reference")
    cohort = cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), reference.patient_nbr.to_numpy()) or not np.array_equal(
        cohort.target.to_numpy(), reference.y_true.to_numpy()
    ):
        raise ValueError("Raw CatBoost patients or labels differ from reference")
    raw, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    if len(raw.columns) != 46:
        raise ValueError("Expected the original 46 CatBoost discharge predictors")
    ccs_columns = [f"ccs_{column}" for column in DIAG]
    frame = pd.concat([raw.reset_index(drop=True), hospital[ccs_columns].astype(str).reset_index(drop=True)], axis=1)
    ccs_categorical = list(dict.fromkeys([*categorical, *ccs_columns]))

    if args.cdc_teacher == "catboost":
        cdc_score, cdc_audit = cdc_catboost_proxy(args, hospital)
    else:
        cdc_score, cdc_audit = cdc_proxy(args, hospital)
    frame["cdc_poor_health"] = cdc_score
    cms_table, lookup = read_cms(args.cms_prior)
    frame["cms_readmission_rate"] = hospital.ccs_diag_1.map(lookup).fillna(-1.).to_numpy(dtype=float)
    frame["cms_mapped"] = hospital.ccs_diag_1.isin(lookup).to_numpy(dtype=int)
    if "uci_principal_diagnosis_rows" in cms_table and int(frame.cms_mapped.sum()) != int(
        cms_table.uci_principal_diagnosis_rows.sum()
    ):
        raise ValueError("CMS coverage differs from the saved prior")
    return frame, ccs_categorical, {
        "cdc": cdc_audit, "cms_mapped_rows": int(frame.cms_mapped.sum()),
        "ccs_columns": ccs_columns,
    }


def run(args) -> None:
    if args.seed != 777 or not 1 <= args.max_folds <= 5:
        raise ValueError("Saved OOF inputs require seed 777 and 1..5 folds")
    if args.bootstrap < 0 or args.threads < 1 or not 0 < args.alert_fraction < 1:
        raise ValueError("Invalid bootstrap, threads or alert fraction")
    sources = list(dict.fromkeys(args.sources))
    reference = load_reference(args.reference_oof)
    base = aligned_saved(reference, args.ccs_oof,
                         ["ccs_only_hgb", "catboost_raw_plus_ccs", "ccs_hgb_catboost_raw_plus_ccs_50"])
    raw = aligned_saved(reference, args.paper_oof, ["catboost_raw", "ccs_only_hgb"])
    cms = aligned_saved(reference, args.cms_oof, ["catboost_raw", "cms_catboost_raw", "ccs_only_hgb"])
    cdc = aligned_saved(reference, args.cdc_oof, ["catboost_raw", "cdc"])
    for name in ("catboost_raw", "ccs_only_hgb"):
        if not np.allclose(raw[name], cms[name], rtol=0, atol=1e-12):
            raise ValueError(f"Saved raw/CMS models disagree: {name}")
    if not np.allclose(raw.catboost_raw, cdc.catboost_raw, rtol=0, atol=1e-12):
        raise ValueError("Saved raw/CDC CatBoost baselines disagree")
    if not np.allclose(base.ccs_only_hgb, raw.ccs_only_hgb, rtol=0, atol=1e-12):
        raise ValueError("Saved CCS HGB baselines disagree")
    if not np.allclose(base.ccs_hgb_catboost_raw_plus_ccs_50,
                       (base.ccs_only_hgb + base.catboost_raw_plus_ccs) / 2,
                       rtol=0, atol=1e-12):
        raise ValueError("Saved CCS 50/50 ensemble is inconsistent")
    choices = chosen_ccs_parameters(args.ccs_trials)
    raw_choices = selected_catboost(args.paper_trials) if args.cdc_teacher == "catboost" and "cdc" in sources else None
    frame, categorical, feature_audit = make_features(args, reference)
    folds = sorted(reference.Fold.unique())[:args.max_folds]
    selected = reference.Fold.isin(folds).to_numpy()
    oof = reference.loc[selected, KEYS].copy().reset_index(drop=True)
    hgb = base.loc[selected, "ccs_only_hgb"].to_numpy(dtype=float)
    oof["ccs_hgb_raw_catboost_50"] = (hgb + raw.loc[selected, "catboost_raw"].to_numpy(dtype=float)) / 2
    oof["ccs_hgb_ccs_catboost_50"] = base.loc[selected, "ccs_hgb_catboost_raw_plus_ccs_50"].to_numpy(dtype=float)
    if "cdc" in sources:
        saved_cdc = (hgb + cdc.loc[selected, "cdc"].to_numpy(dtype=float)) / 2
        historical_name = "ccs_hgb_raw_catboost_cdc_50" if args.cdc_teacher == "xgboost" else "ccs_hgb_raw_catboost_cdc_xgbteacher_50"
        oof[historical_name] = saved_cdc
    if "cms" in sources:
        oof["ccs_hgb_raw_catboost_cms_50"] = (hgb + cms.loc[selected, "cms_catboost_raw"].to_numpy(dtype=float)) / 2
    if args.audit_only:
        print(f"[Audit] rows={len(reference):,} patients={reference.patient_nbr.nunique():,} "
              f"folds={folds} CMS_mapped={feature_audit['cms_mapped_rows']:,}", flush=True)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    new_columns = {"cdc": ["cdc_poor_health"], "cms": ["cms_readmission_rate", "cms_mapped"]}
    for fold in folds:
        train = reference.Fold.ne(fold).to_numpy()
        test = reference.Fold.eq(fold).to_numpy()
        output_test = oof.Fold.eq(fold).to_numpy()
        for source in sources:
            columns = frame.columns[:49].tolist() + new_columns[source]
            if source == "cdc" and args.cdc_teacher == "catboost":
                raw_columns = frame.columns[:46].tolist() + new_columns[source]
                _, raw_probability = fit_catboost(
                    frame.loc[train, raw_columns].reset_index(drop=True),
                    reference.loc[train, "y_true"].to_numpy(dtype=int),
                    frame.loc[test, raw_columns].reset_index(drop=True),
                    [name for name in categorical if name in raw_columns],
                    raw_choices.loc[fold], args,
                )
                oof.loc[output_test, "ccs_hgb_raw_catboost_cdc_cbteacher_50"] = (
                    hgb[output_test] + raw_probability
                ) / 2
            _, probability = fit_catboost(
                frame.loc[train, columns].reset_index(drop=True),
                reference.loc[train, "y_true"].to_numpy(dtype=int),
                frame.loc[test, columns].reset_index(drop=True),
                categorical, choices.loc[fold], args,
            )
            name = (f"ccs_hgb_ccs_catboost_cdc_cbteacher_50" if source == "cdc" and args.cdc_teacher == "catboost"
                    else f"ccs_hgb_ccs_catboost_{source}_50")
            oof.loc[output_test, name] = (hgb[output_test] + probability) / 2
            print(f"[Fold {fold}] CCS CatBoost + {source.upper()} "
                  f"depth={int(choices.loc[fold, 'Depth'])} "
                  f"trees={int(choices.loc[fold, 'Trees'])}", flush=True)
        oof.loc[output_test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)

    methods = ["ccs_hgb_raw_catboost_50", "ccs_hgb_ccs_catboost_50"]
    if "cdc" in sources:
        methods.append(historical_name)
        if args.cdc_teacher == "catboost":
            methods.extend(["ccs_hgb_raw_catboost_cdc_cbteacher_50",
                            "ccs_hgb_ccs_catboost_cdc_cbteacher_50"])
        else:
            methods.append("ccs_hgb_ccs_catboost_cdc_50")
    if "cms" in sources:
        methods.extend(["ccs_hgb_raw_catboost_cms_50", "ccs_hgb_ccs_catboost_cms_50"])
    if oof[methods].isna().any().any():
        raise ValueError("At least one outer-test prediction is missing")
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    summarize(oof, methods, args.outdir, args.alert_fraction)
    comparisons = []
    for source in sources:
        for kind in ("raw", "ccs"):
            treatment = (f"ccs_hgb_{kind}_catboost_cdc_cbteacher_50"
                         if source == "cdc" and args.cdc_teacher == "catboost"
                         else f"ccs_hgb_{kind}_catboost_{source}_50")
            comparisons.append((treatment, f"ccs_hgb_{kind}_catboost_50"))
    if "cdc" in sources and args.cdc_teacher == "catboost":
        comparisons.append(("ccs_hgb_raw_catboost_cdc_cbteacher_50", historical_name))
    paired_patient_bootstrap(oof, comparisons, args.outdir, args.bootstrap, args.seed)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "folds": [int(fold) for fold in folds],
        "reference_oof": str(args.reference_oof), "paper_oof": str(args.paper_oof),
        "ccs_oof": str(args.ccs_oof), "cdc_oof": str(args.cdc_oof),
        "cms_oof": str(args.cms_oof), "ccs_trials": str(args.ccs_trials),
        "paper_trials": str(args.paper_trials), "sources": sources,
        "cdc_teacher": args.cdc_teacher,
        "features": {source: frame.columns[:49].tolist() + new_columns[source] for source in sources},
        "feature_audit": feature_audit,
        "fit": "saved train-only raw/CCS CatBoost parameters; refit on each outer training fold",
        "fusion": "0.5 * saved CCS HGB + 0.5 * CatBoost",
        "alert_fraction": args.alert_fraction, "bootstrap": args.bootstrap,
    }, indent=2) + "\n")
    print(f"Saved {len(methods)}-model comparison to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--cdc", type=Path, default=ROOT / "data/raw/cdc/diabetes_012_health_indicators_BRFSS2015.csv")
    parser.add_argument("--cms-prior", type=Path, default=ROOT / "data/processed/cms/cms_readmission_ccs_prior.csv")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--paper-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--ccs-oof", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--cdc-oof", type=Path, default=ROOT / "results/cms_meps_cdc_catboost_seed777/oof_all.csv.gz")
    parser.add_argument("--cms-oof", type=Path, default=ROOT / "results/cms_group_prior_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--ccs-trials", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/catboost_inner_trials.csv")
    parser.add_argument("--paper-trials", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/catboost_inner_trials.csv")
    parser.add_argument("--cdc-teacher", choices=("xgboost", "catboost"), default="xgboost")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.outdir is None:
        directory = "ccs_cdc_catboost_teacher_5fold_seed777" if args.cdc_teacher == "catboost" else "ccs_external_catboost_5fold_seed777"
        args.outdir = ROOT / "results" / directory
    run(args)


if __name__ == "__main__":
    main()
