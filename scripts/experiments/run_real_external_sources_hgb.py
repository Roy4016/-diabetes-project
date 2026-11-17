#!/usr/bin/env python3
"""Compare real MEPS 2008-2024 and NHANES 2015-2018 features on CCS-HGB."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from _paths import CCS_RAW, PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from diabetes_project.harmonized_meps import (
    PROXIES, RACES, fit_domain_weighted_proxies, hospital_bridge, load_longitudinal_pairs,
)

from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import choose_hgb, encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs


NHANES_FEATURES = ["age", "male"] + [f"race_{name}" for name in RACES]
VARIANTS = {
    "ccs_meps_extended_hgb": list(PROXIES),
    "ccs_nhanes_hgb": ["nhanes_hospitalization"],
    "ccs_meps_nhanes_hgb": list(PROXIES) + ["nhanes_hospitalization"],
}


def nhanes_proxy(nhanes):
    if len(nhanes) < 100 or nhanes.hospitalized_past_year.nunique() != 2:
        raise ValueError("NHANES diabetic hospitalization cohort is too small or one-class")
    source = pd.DataFrame(index=nhanes.index)
    source["age"] = pd.to_numeric(nhanes.age, errors="coerce").fillna(65).clip(0, 100) / 100
    source["male"] = pd.to_numeric(nhanes.male, errors="coerce").fillna(.5)
    for race in RACES:
        source[f"race_{race}"] = nhanes.race_common.eq(race).astype(float)
    weights = pd.to_numeric(nhanes.person_weight, errors="coerce").to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.)
    weights = weights / weights.mean()
    model = LogisticRegression(C=.1, max_iter=2000)
    model.fit(source[NHANES_FEATURES], nhanes.hospitalized_past_year, sample_weight=weights)
    return model


def align_reference(oof, path):
    reference = pd.read_csv(path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_ccs_only_hgb",
    ])
    if reference.encounter_id.duplicated().any() or oof.encounter_id.duplicated().any():
        raise ValueError("OOF encounter identifiers must be unique")
    aligned = oof.merge(reference, on="encounter_id", how="left", validate="one_to_one", suffixes=("", "_ref"))
    if len(aligned) != len(reference) or aligned.p_ccs_only_hgb.isna().any():
        raise ValueError("Reference OOF does not cover all current encounters")
    for name in ("patient_nbr", "Fold", "y_true"):
        if not np.array_equal(aligned[name].to_numpy(), aligned[f"{name}_ref"].to_numpy()):
            raise ValueError(f"Reference OOF {name} mismatch")
    return aligned.drop(columns=[f"{name}_ref" for name in ("patient_nbr", "Fold", "y_true")])


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    base_columns = raw_discharge_predictors(hospital)
    ccs_map, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, ccs_map)
    pairs = load_longitudinal_pairs(args.meps)
    nhanes = pd.read_csv(args.nhanes)
    nhanes_model = nhanes_proxy(nhanes)
    for name, extra in VARIANTS.items():
        if not extra:
            raise ValueError(f"Empty feature variant: {name}")
    shared_columns = base_columns + [f"ccs_{name}" for name in DIAG]
    print(f"[External] MEPS consecutive-year pairs={len(pairs):,}; NHANES diabetic respondents={len(nhanes):,}", flush=True)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_parts, choices, diagnostics = [], [], []
    for fold, (train_ids, test_ids) in enumerate(
        splitter.split(hospital, hospital.target, groups=hospital.patient_nbr), 1,
    ):
        train, test = hospital.iloc[train_ids], hospital.iloc[test_ids]
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient leakage in outer split")
        enriched, info = fit_domain_weighted_proxies(
            pairs, train, {"train": train, "test": test}, args.seed + fold * 10000,
            weight_clip=(args.domain_weight_min, args.domain_weight_max),
        )
        train, test = enriched["train"], enriched["test"]
        for frame in (train, test):
            x = hospital_bridge(frame)[NHANES_FEATURES]
            frame["nhanes_hospitalization"] = nhanes_model.predict_proba(x)[:, 1].astype(np.float32)
        info["Fold"] = fold
        diagnostics.append(info)

        inner_ids, val_ids = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(train, train.target, groups=train.patient_nbr))
        inner, val = train.iloc[inner_ids], train.iloc[val_ids]
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        for name, external_columns in VARIANTS.items():
            columns = shared_columns + external_columns
            x_inner, x_val, _ = encode_frames(inner, val, val, columns, args.min_frequency)
            val_ap, params, iterations, _ = choose_hgb(
                x_inner, x_val, inner.target.to_numpy(), val.target.to_numpy(), args.seed + fold,
            )
            x_train, _, x_test = encode_frames(train, train, test, columns, args.min_frequency)
            model = hgb_model(
                args.seed + fold, {**params, "max_iter": iterations}, early_stopping=False,
            ).fit(x_train, train.target.to_numpy())
            probability = model.predict_proba(x_test)[:, 1]
            part[f"p_{name}"] = probability
            choices.append({"Fold": fold, "Experiment": name, "ValidationAP": val_ap,
                            "Iterations": iterations, "Parameters": json.dumps(params)})
            print(f"[Fold {fold}] {name} PR_AUC={average_precision_score(test.target, probability):.6f}", flush=True)
        oof_parts.append(part)
        pd.concat(oof_parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)

    oof = align_reference(pd.concat(oof_parts, ignore_index=True), args.reference_oof)
    models = ["ccs_only_hgb", *VARIANTS]
    rows = [
        {"Fold": int(fold), "Experiment": name,
         "PR_AUC_AP": average_precision_score(part.y_true, part[f"p_{name}"])}
        for fold, part in oof.groupby("Fold", sort=True) for name in models
    ]
    result = pd.DataFrame(rows)
    summary = result.groupby("Experiment").PR_AUC_AP.agg(["mean", "std"]).reset_index()
    comparisons = [(name, "ccs_only_hgb") for name in VARIANTS]
    comparisons += [("ccs_meps_nhanes_hgb", "ccs_meps_extended_hgb")]
    bootstrap = cluster_bootstrap(oof, models, comparisons, args.bootstrap, args.seed + 17000)
    oof.to_csv(output / "oof_all.csv", index=False)
    result.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "selected_hgb.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "meps_domain_diagnostics.csv", index=False)
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default="data/processed/meps_2008_2024/meps_diabetes_bridge8.csv")
    parser.add_argument("--nhanes", default="data/processed/nhanes/diabetes_hospitalization_2015_2018.csv")
    parser.add_argument("--ccs-file", default=CCS_RAW)
    parser.add_argument("--reference-oof", type=Path, required=True)
    parser.add_argument("--outdir", default=result_dir("meps_extended_nhanes_hgb"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--domain-weight-min", type=float, default=.1)
    parser.add_argument("--domain-weight-max", type=float, default=10.)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 1 or args.min_frequency < 1:
        parser.error("folds >= 2, bootstrap >= 1, min-frequency >= 1 are required")
    for path in (args.meps, args.nhanes, args.reference_oof):
        if not Path(path).is_file():
            parser.error(f"Required input file not found: {path}")
    run(args)


if __name__ == "__main__":
    main()
