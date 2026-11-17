#!/usr/bin/env python3
"""Evaluate HGB with domain-weighted longitudinal MEPS features on held-out patients."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from _paths import MEPS_BRIDGE, PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from diabetes_project.harmonized_meps import PROXIES, fit_domain_weighted_proxies, load_longitudinal_pairs

from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import choose_hgb, encode_frames, hgb_model


def reference_oof(path, predictions):
    reference = pd.read_csv(path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_hgb_tuned", "p_ensemble_fixed_50",
    ])
    if reference.encounter_id.duplicated().any() or predictions.encounter_id.duplicated().any():
        raise ValueError("Reference and new OOF must have unique encounter_id values")
    aligned = predictions.merge(
        reference, on="encounter_id", how="left", validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if len(aligned) != len(reference) or aligned.p_hgb_tuned.isna().any():
        raise ValueError("Reference OOF does not cover the new OOF encounters")
    for name in ("patient_nbr", "Fold", "y_true"):
        if not np.array_equal(aligned[name].to_numpy(), aligned[f"{name}_reference"].to_numpy()):
            raise ValueError(f"Reference OOF {name} does not match the new split")
    return aligned.drop(columns=[f"{name}_reference" for name in ("patient_nbr", "Fold", "y_true")])


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    pairs = load_longitudinal_pairs(args.meps)
    columns = raw_discharge_predictors(hospital) + list(PROXIES)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_parts, choices, diagnostics = [], [], []

    for fold, (train_ids, test_ids) in enumerate(
        splitter.split(hospital, hospital.target, groups=hospital.patient_nbr), 1,
    ):
        train = hospital.iloc[train_ids].copy()
        test = hospital.iloc[test_ids].copy()
        if set(train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient leakage in outer split")

        # Source-to-target weights see only this fold's hospital training rows.
        enriched, info = fit_domain_weighted_proxies(
            pairs, train, {"train": train, "test": test},
            args.seed + fold * 10000,
            weight_clip=(args.domain_weight_min, args.domain_weight_max),
        )
        train, test = enriched["train"], enriched["test"]
        info["Fold"] = fold
        diagnostics.append(info)

        inner_ids, val_ids = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(train, train.target, groups=train.patient_nbr))
        inner, val = train.iloc[inner_ids], train.iloc[val_ids]
        if inner.target.nunique() != 2 or val.target.nunique() != 2:
            raise ValueError(f"One-class inner partition in fold {fold}")
        x_inner, x_val, _ = encode_frames(inner, val, val, columns, args.min_frequency)
        val_ap, parameters, iterations, _ = choose_hgb(
            x_inner, x_val, inner.target.to_numpy(), val.target.to_numpy(), args.seed + fold,
        )
        x_train, _, x_test = encode_frames(train, train, test, columns, args.min_frequency)
        model = hgb_model(
            args.seed + fold, {**parameters, "max_iter": iterations}, early_stopping=False,
        ).fit(x_train, train.target.to_numpy())
        probability = model.predict_proba(x_test)[:, 1]

        part = test[["encounter_id", "patient_nbr", "target"]].rename(
            columns={"target": "y_true"},
        ).copy()
        part["Fold"] = fold
        part["p_hgb_meps_augmented"] = probability
        oof_parts.append(part)
        choices.append({
            "Fold": fold, "ValidationAP": val_ap, "Iterations": iterations,
            "Parameters": json.dumps(parameters, sort_keys=True),
        })
        print(
            f"[Fold {fold}] augmented_AP={average_precision_score(test.target, probability):.6f} "
            f"validation_AP={val_ap:.6f} domain_ESS={info['DomainWeightESS']:.0f}",
            flush=True,
        )
        pd.concat(oof_parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)

    oof = reference_oof(args.reference_oof, pd.concat(oof_parts, ignore_index=True))
    models = ("hgb_tuned", "ensemble_fixed_50", "hgb_meps_augmented")
    metrics = []
    for fold, part in oof.groupby("Fold", sort=True):
        for name in models:
            probability = part[f"p_{name}"].to_numpy()
            metrics.append({
                "Fold": int(fold), "Experiment": name,
                "PR_AUC_AP": average_precision_score(part.y_true, probability),
                "AUROC": roc_auc_score(part.y_true, probability),
                "Brier": brier_score_loss(part.y_true, probability),
            })
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby("Experiment").agg(
        PR_AUC_mean=("PR_AUC_AP", "mean"), PR_AUC_std=("PR_AUC_AP", "std"),
        AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"),
    ).reset_index().sort_values("PR_AUC_mean", ascending=False)
    bootstrap = cluster_bootstrap(
        oof, list(models), [
            ("hgb_meps_augmented", "hgb_tuned"),
            ("hgb_meps_augmented", "ensemble_fixed_50"),
        ], args.bootstrap, args.seed + 17000,
    )
    oof.to_csv(output / "oof_all.csv", index=False)
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "selected_hgb.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "domain_weighting_diagnostics.csv", index=False)
    config = vars(args).copy()
    config["reference_oof"] = str(args.reference_oof)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nFold-stratified patient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default=MEPS_BRIDGE)
    parser.add_argument("--reference-oof", required=True, type=Path)
    parser.add_argument("--outdir", default=result_dir("meps_augmented_hgb"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--domain-weight-min", type=float, default=.1)
    parser.add_argument("--domain-weight-max", type=float, default=10.)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if (args.folds < 2 or args.bootstrap < 1 or args.min_frequency < 1
            or args.domain_weight_min <= 0 or args.domain_weight_max < args.domain_weight_min):
        parser.error("Invalid folds, bootstrap, frequency, or domain weight bounds")
    if not args.reference_oof.is_file():
        parser.error(f"Reference OOF not found: {args.reference_oof}")
    run(args)


if __name__ == "__main__":
    main()
