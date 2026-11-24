#!/usr/bin/env python3
"""Compare CMS, MEPS and CDC features on the fixed patient-disjoint UCI folds.

CMS supplies four national condition readmission rates; MEPS supplies observed
30-day event-CCS rates; CDC supplies a poor-self-rated-health proxy trained only
on diabetic BRFSS respondents. CDC's outcome is not hospital readmission.
The saved raw-ICD CatBoost hyperparameters and outer folds are held fixed.
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
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import (  # noqa: E402
    catboost_shap_sample, load_comparison_cohort, make_paper_catboost_features,
)
from run_cdc_xgb import BRIDGE_COLS, load_cdc  # noqa: E402
from run_cms_group_prior_5fold import checksum, read_cms, selected_catboost  # noqa: E402
from run_dual_external_meps_nhanes import hospital_frame, load_reference  # noqa: E402
from run_meps_ccs_group_hgb import (  # noqa: E402
    EVENT_STATS, attach_group_features, make_event_group_rates,
)
from run_paper_fair_comparison import (  # noqa: E402
    fit_catboost, paired_patient_bootstrap, save_shap, summarize,
)

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
FEATURES = {
    "meps": list(EVENT_STATS),
    "cdc": ["cdc_poor_health"],
    "cms_meps": ["cms_readmission_rate", "cms_mapped", *EVENT_STATS],
    "cms_cdc": ["cms_readmission_rate", "cms_mapped", "cdc_poor_health"],
    "meps_cdc": [*EVENT_STATS, "cdc_poor_health"],
    "cms_meps_cdc": ["cms_readmission_rate", "cms_mapped", *EVENT_STATS, "cdc_poor_health"],
}
BASE = ["catboost_raw", "cms_catboost_raw", "ccs_only_hgb_catboost_raw_50"]


def aligned_saved(reference: pd.DataFrame, path: Path, columns: list[str]) -> pd.DataFrame:
    saved = pd.read_csv(path, usecols=KEYS + columns)
    if len(saved) != len(reference) or saved.encounter_id.duplicated().any():
        raise ValueError(f"Saved OOF cohort differs: {path}")
    saved = saved.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(saved[KEYS].to_numpy(), reference[KEYS].to_numpy()):
        raise ValueError(f"Saved OOF split, patient or label differs: {path}")
    if saved[columns].isna().any().any():
        raise ValueError(f"Saved OOF has missing scores: {path}")
    return saved


def cdc_proxy(args, hospital: pd.DataFrame) -> tuple[np.ndarray, dict]:
    source = load_cdc(args.cdc)
    source = source.loc[source.Diabetes_012.eq(2)].copy()
    if len(source) < 1000 or source.poor_health.nunique() != 2:
        raise ValueError("Too few labelled CDC diabetic respondents")
    if hospital[BRIDGE_COLS].isna().all(axis=1).any():
        raise ValueError("A UCI encounter has no CDC bridge fields")
    x = source[BRIDGE_COLS].astype(float)
    y = source.poor_health.astype(int)
    train, validation = train_test_split(
        np.arange(len(source)), test_size=.20, stratify=y, random_state=args.seed,
    )

    def model(random_state: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=.05,
            subsample=.85, colsample_bytree=1., min_child_weight=5,
            reg_lambda=2., objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu", random_state=random_state,
            n_jobs=args.threads,
        )

    check = model(args.seed).fit(x.iloc[train], y.iloc[train])
    validation_score = check.predict_proba(x.iloc[validation])[:, 1]
    final = model(args.seed).fit(x, y)
    score = final.predict_proba(hospital[BRIDGE_COLS].astype(float))[:, 1]
    if not np.isfinite(score).all():
        raise ValueError("CDC proxy produced nonfinite scores")
    audit = {
        "respondents": int(len(source)), "poor_health_prevalence": float(y.mean()),
        "held_out_cdc_ap": float(average_precision_score(y.iloc[validation], validation_score)),
        "held_out_cdc_auroc": float(roc_auc_score(y.iloc[validation], validation_score)),
        "bridge_features": list(BRIDGE_COLS),
        "target": "self-rated general health fair or poor; NOT readmission",
    }
    print(f"[CDC] diabetic respondents={len(source):,}; validation poor-health "
          f"AP={audit['held_out_cdc_ap']:.6f}; AUROC={audit['held_out_cdc_auroc']:.6f}", flush=True)
    return score, audit


def build_features(args, reference: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    hospital, _ = hospital_frame(args, reference)
    cms_table, lookup = read_cms(args.cms_prior)
    hospital["cms_readmission_rate"] = hospital.ccs_diag_1.map(lookup).fillna(-1.).astype(float)
    hospital["cms_mapped"] = hospital.ccs_diag_1.isin(lookup).astype(int)
    if "uci_principal_diagnosis_rows" in cms_table and int(hospital.cms_mapped.sum()) != int(
        cms_table.uci_principal_diagnosis_rows.sum()
    ):
        raise ValueError("CMS coverage changed since the prior was prepared")

    events = pd.read_csv(args.meps_events, dtype={"person_key": str, "event_id": str})
    event_ccs = pd.read_csv(args.meps_event_ccs, dtype={"person_key": str, "event_id": str})
    rates, global_rates, n_events, n_linked = make_event_group_rates(
        events, event_ccs, args.min_group, args.alpha,
    )
    hospital = attach_group_features(hospital, rates, global_rates, EVENT_STATS)
    if not np.isfinite(hospital[list(EVENT_STATS)].to_numpy(dtype=float)).all():
        raise ValueError("MEPS event rates are not finite")

    cdc_score, cdc_audit = cdc_proxy(args, hospital)
    hospital["cdc_poor_health"] = cdc_score
    cohort = load_comparison_cohort(args.hospital)
    if len(cohort) != len(reference) or cohort.encounter_id.duplicated().any():
        raise ValueError("CatBoost cohort differs from reference")
    cohort = cohort.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if not np.array_equal(cohort.patient_nbr.to_numpy(), reference.patient_nbr.to_numpy()) or not np.array_equal(
        cohort.target.to_numpy(), reference.y_true.to_numpy()
    ):
        raise ValueError("CatBoost patient or label mismatch")
    features, categorical = make_paper_catboost_features(cohort, raw_icd=True)
    external_columns = sorted({name for columns in FEATURES.values() for name in columns})
    features[external_columns] = hospital[external_columns].to_numpy()
    audit = {
        "cms_mapped_rows": int(hospital.cms_mapped.sum()),
        "cms_mapped_fraction": float(hospital.cms_mapped.mean()),
        "cms_groups": cms_table.ccs.astype(str).tolist(),
        "meps_events": int(n_events), "meps_linked_ccs": int(n_linked),
        "meps_groups": int(len(rates)), "meps_global_readmit_30d": float(global_rates["next_er"]),
        "meps_global_er_revisit_30d": float(global_rates["next_ip"]),
        "cdc": cdc_audit,
    }
    print(f"[External] CMS matched={audit['cms_mapped_rows']:,}/{len(reference):,}; "
          f"MEPS events={n_events:,}, linked CCS={n_linked:,}, groups={len(rates)}", flush=True)
    return features, categorical, audit


def run(args) -> None:
    if not 1 <= args.max_folds <= 5 or not 0 < args.alert_fraction < 1:
        raise ValueError("Use 1..5 folds and 0<alert-fraction<1")
    if args.shap_sample < 0 or args.bootstrap < 0 or args.threads < 1:
        raise ValueError("Invalid SHAP, bootstrap or thread count")
    variants = list(dict.fromkeys(args.variants))
    if not variants or any(name not in FEATURES for name in variants):
        raise ValueError("Unknown or empty feature variant")
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference_oof)
    paper = aligned_saved(reference, args.catboost_oof, ["catboost_raw", "ccs_only_hgb"])
    cms = aligned_saved(reference, args.cms_oof, ["catboost_raw", "cms_catboost_raw", "ccs_only_hgb"])
    for name in ("catboost_raw", "ccs_only_hgb"):
        if not np.allclose(paper[name].to_numpy(), cms[name].to_numpy(), atol=1e-12):
            raise ValueError(f"Saved CMS and paper baselines differ: {name}")
    if not np.allclose(reference.p_ccs_only_hgb.to_numpy(), paper.ccs_only_hgb.to_numpy(), atol=1e-12):
        raise ValueError("HGB baseline differs from reference")
    choices = selected_catboost(args.catboost_trials)
    feature_matrix, categorical, external_audit = build_features(args, reference)
    folds = sorted(reference.Fold.unique())[:args.max_folds]
    selected = reference.Fold.isin(folds).to_numpy()
    oof = reference.loc[selected, KEYS].copy().reset_index(drop=True)
    oof["catboost_raw"] = paper.loc[selected, "catboost_raw"].to_numpy()
    oof["cms_catboost_raw"] = cms.loc[selected, "cms_catboost_raw"].to_numpy()
    oof["ccs_only_hgb_catboost_raw_50"] = (
        reference.loc[selected, "p_ccs_only_hgb"].to_numpy() + oof.catboost_raw.to_numpy()
    ) / 2
    shap_tables = []
    for fold in folds:
        train = reference.Fold.ne(fold).to_numpy()
        test = reference.Fold.eq(fold).to_numpy()
        oof_test = oof.Fold.eq(fold).to_numpy()
        for variant in variants:
            # The first 46 columns are the original raw-discharge predictors.
            columns = feature_matrix.columns[:46].tolist() + FEATURES[variant]
            x_train = feature_matrix.loc[train, columns].reset_index(drop=True)
            x_test = feature_matrix.loc[test, columns].reset_index(drop=True)
            model, score = fit_catboost(
                x_train, reference.loc[train, "y_true"].to_numpy(dtype=int),
                x_test, categorical, choices.loc[fold], args,
            )
            oof.loc[oof_test, variant] = score
            print(f"[Fold {fold}] {variant} PR_AUC="
                  f"{average_precision_score(reference.loc[test, 'y_true'], score):.6f}", flush=True)
            if variant == "cms_meps_cdc" and args.shap_sample:
                shap = catboost_shap_sample(model, x_test, categorical, args.shap_sample,
                                            args.seed + 100 * int(fold))
                shap.insert(0, "Fold", int(fold))
                shap.insert(0, "Model", "cms_meps_cdc")
                shap_tables.append(shap)
        if "cms_meps_cdc" in variants:
            oof.loc[oof_test, "cms_meps_cdc_hgb50"] = (
                oof.loc[oof_test, "cms_meps_cdc"].to_numpy()
                + reference.loc[test, "p_ccs_only_hgb"].to_numpy()
            ) / 2
        oof.loc[oof_test].to_csv(args.outdir / f"oof_fold_{fold}.csv.gz", index=False)

    methods = BASE + variants + (["cms_meps_cdc_hgb50"] if "cms_meps_cdc" in variants else [])
    if oof[methods].isna().any().any():
        raise ValueError("A prediction was not generated")
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    summary = summarize(oof, methods, args.outdir, args.alert_fraction)
    comparisons = [("cms_catboost_raw", "catboost_raw")]
    for variant in variants:
        comparisons.append((variant, "catboost_raw"))
    if "cms_meps_cdc" in variants:
        comparisons += [
            ("cms_meps_cdc_hgb50", "ccs_only_hgb_catboost_raw_50"),
            ("cms_meps_cdc", "cms_catboost_raw"),
        ]
        comparisons += [("cms_meps_cdc", name) for name in variants if name != "cms_meps_cdc"]
    paired_patient_bootstrap(oof, comparisons, args.outdir, args.bootstrap, args.seed)
    save_shap(shap_tables, args.outdir)
    audit = {
        "reference_rows": int(len(reference)), "reference_patients": int(reference.patient_nbr.nunique()),
        "completed_folds": [int(fold) for fold in folds], "variants": variants,
        "patient_disjoint": bool(reference.groupby("patient_nbr").Fold.nunique().max() == 1),
        "feature_sets": FEATURES, "external": external_audit,
        "selected_catboost": "original raw-ICD inner-validation choices held fixed in each fold",
        "alert_policy": f"top {args.alert_fraction:.1%} scores within each fold",
        "source_sha256": {name: checksum(path) for name, path in {
            "cms": args.cms_prior, "meps_events": args.meps_events,
            "meps_event_ccs": args.meps_event_ccs, "cdc": args.cdc,
            "reference_oof": args.reference_oof, "catboost_oof": args.catboost_oof,
            "cms_oof": args.cms_oof,
        }.items()},
    }
    (args.outdir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(summary)} methods to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--cms-prior", type=Path, default=ROOT / "data/processed/cms/cms_readmission_ccs_prior.csv")
    parser.add_argument("--meps-events", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--meps-event-ccs", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--cdc", type=Path, default=ROOT / "data/raw/cdc/diabetes_012_health_indicators_BRFSS2015.csv")
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/meps_event_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--catboost-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--catboost-trials", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/catboost_inner_trials.csv")
    parser.add_argument("--cms-oof", type=Path, default=ROOT / "results/cms_group_prior_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/cms_meps_cdc_catboost_seed777")
    parser.add_argument("--variants", nargs="+", choices=sorted(FEATURES), default=list(FEATURES))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=777)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
