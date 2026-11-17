#!/usr/bin/env python3
"""Cross-fitted MEPS residual correction on top of a discharge-time HGB model.

The outer split is patient-disjoint. Within every outer-training fold, HGB
out-of-fold probabilities are generated with patient-disjoint cross-fitting.
A small L2-regularized logistic correction then uses only the external MEPS
ER/IP proxies and their interactions with HGB risk. The uncorrected HGB is an
explicit validation candidate, so the method can fall back to the baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from _paths import MEPS_BRIDGE, PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.harmonized_meps import (
    PROXIES as HARMONIZED_PROXIES,
    fit_domain_weighted_proxies,
    load_longitudinal_pairs,
)
from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import external_meps, load_meps, prepare_hospital
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import choose_hgb, encode_frames, hgb_model


MODELS = ["hgb_tuned", "meps_residual"]
COMPARISONS = [("meps_residual", "hgb_tuned")]


def clipped_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def correction_features(base_probability, er_probability, ip_probability):
    """MEPS-only correction terms; baseline logit is supplied as a fixed offset."""
    base = clipped_logit(base_probability)
    er = clipped_logit(er_probability)
    ip = clipped_logit(ip_probability)
    return np.column_stack([
        er,
        ip,
        base * er,
        base * ip,
        er * ip,
    ])


def fit_offset_logistic(features, labels, offset, l2):
    """Fit sigmoid(offset + intercept + features @ beta) with L2 shrinkage."""
    x = np.column_stack([np.ones(len(features)), features]).astype(float, copy=False)
    y = np.asarray(labels, dtype=float)
    offset = np.asarray(offset, dtype=float)

    def objective(parameters):
        score = offset + x @ parameters
        probability = expit(score)
        loss = np.mean(np.logaddexp(0.0, score) - y * score)
        penalty = 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
        gradient = x.T @ (probability - y) / len(y)
        gradient[1:] += l2 * parameters[1:]
        return loss + penalty, gradient

    result = minimize(
        objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Residual correction optimizer failed: {result.message}")
    return result.x


def predict_offset_logistic(features, offset, parameters):
    x = np.column_stack([np.ones(len(features)), features]).astype(float, copy=False)
    return expit(np.asarray(offset, dtype=float) + x @ parameters)


def cross_fitted_hgb(frame, columns, params, iterations, folds, seed, min_frequency):
    probability = np.full(len(frame), np.nan, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for cross_fold, (train_ids, holdout_ids) in enumerate(
        splitter.split(frame, frame.target, groups=frame.patient_nbr), 1,
    ):
        train, holdout = frame.iloc[train_ids], frame.iloc[holdout_ids]
        if set(train.patient_nbr).intersection(holdout.patient_nbr):
            raise ValueError("Patient leakage in HGB cross-fitting")
        x_train, _, x_holdout = encode_frames(
            train, train, holdout, columns, min_frequency,
        )
        model = hgb_model(
            seed + cross_fold, {**params, "max_iter": iterations}, early_stopping=False,
        ).fit(x_train, train.target.to_numpy())
        probability[holdout_ids] = model.predict_proba(x_holdout)[:, 1]
        print(
            f"    [Cross-fit {cross_fold}/{folds}] rows={len(holdout):,} "
            f"PR_AUC={average_precision_score(holdout.target, probability[holdout_ids]):.6f}",
            flush=True,
        )
    if np.isnan(probability).any():
        raise RuntimeError("Cross-fitting did not produce every training prediction")
    return probability


def select_correction(frame, base_oof, proxy_columns, l2_values, seed):
    train_ids, val_ids = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=seed,
    ).split(frame, frame.target, groups=frame.patient_nbr))
    train, val = frame.iloc[train_ids], frame.iloc[val_ids]
    raw = correction_features(
        base_oof, frame[proxy_columns[0]], frame[proxy_columns[1]],
    )
    scaler = StandardScaler().fit(raw[train_ids])
    x_train, x_val = scaler.transform(raw[train_ids]), scaler.transform(raw[val_ids])
    train_offset, val_offset = clipped_logit(base_oof[train_ids]), clipped_logit(base_oof[val_ids])
    y_train, y_val = train.target.to_numpy(), val.target.to_numpy()

    candidates = [(average_precision_score(y_val, base_oof[val_ids]), None, None)]
    print(f"    [Correction fallback] val_PR_AUC={candidates[0][0]:.6f}", flush=True)
    for l2 in l2_values:
        parameters = fit_offset_logistic(x_train, y_train, train_offset, l2)
        probability = predict_offset_logistic(x_val, val_offset, parameters)
        ap = average_precision_score(y_val, probability)
        candidates.append((ap, float(l2), parameters))
        print(f"    [Correction L2={l2:g}] val_PR_AUC={ap:.6f}", flush=True)
    return max(candidates, key=lambda candidate: candidate[0])


def attach_reference(oof, reference_path):
    if reference_path is None:
        return oof, []
    reference = pd.read_csv(reference_path, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_ensemble_fixed_50",
    ])
    if reference.encounter_id.duplicated().any() or oof.encounter_id.duplicated().any():
        raise ValueError("encounter_id must be unique before reference OOF alignment")
    renamed = reference.rename(columns={
        "patient_nbr": "reference_patient", "Fold": "reference_fold",
        "y_true": "reference_y",
    })
    merged = oof.merge(renamed, on="encounter_id", how="left", validate="one_to_one")
    if merged.p_ensemble_fixed_50.isna().any():
        raise ValueError("Reference OOF does not contain every current outer-test encounter")
    if not (
        np.array_equal(merged.patient_nbr.to_numpy(), merged.reference_patient.to_numpy())
        and np.array_equal(merged.Fold.to_numpy(), merged.reference_fold.to_numpy())
        and np.array_equal(merged.y_true.to_numpy(), merged.reference_y.to_numpy())
    ):
        raise ValueError("Reference OOF labels, patients, or folds do not match this experiment")
    return merged.drop(columns=["reference_patient", "reference_fold", "reference_y"]), [
        "ensemble_fixed_50",
    ]


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    frame, _ = prepare_hospital(args.hospital)
    external_diagnostics = []
    if args.external_method == "legacy":
        frame = external_meps(load_meps(args.meps), frame, args.seed, output, device="cpu")
        meps_pairs = None
        proxy_columns = ("meps_er_8", "meps_ip_8")
    else:
        meps_pairs = load_longitudinal_pairs(args.meps)
        proxy_columns = HARMONIZED_PROXIES
        print(
            f"[MEPS] loaded {len(meps_pairs):,} consecutive-year pairs for "
            "harmonized domain weighting",
            flush=True,
        )
    columns = raw_discharge_predictors(frame)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows, oof_parts, selections = [], [], []

    for fold, (outer_train_ids, test_ids) in enumerate(
        splitter.split(frame, frame.target, groups=frame.patient_nbr), 1,
    ):
        outer_train, test = frame.iloc[outer_train_ids].reset_index(drop=True), frame.iloc[test_ids]
        if set(outer_train.patient_nbr).intersection(test.patient_nbr):
            raise ValueError("Patient leakage in outer split")
        if args.external_method == "harmonized_weighted":
            proxy_frames, diagnostic = fit_domain_weighted_proxies(
                meps_pairs, outer_train, {"train": outer_train, "test": test},
                args.seed + fold * 10000,
                weight_clip=(args.domain_weight_min, args.domain_weight_max),
            )
            outer_train, test = proxy_frames["train"], proxy_frames["test"]
            diagnostic["Fold"] = fold
            external_diagnostics.append(diagnostic)
            print(
                f"[Fold {fold}] domain_AUROC={diagnostic['DomainTrainingAUROC']:.4f} "
                f"weight_ESS={diagnostic['DomainWeightESS']:.1f}/{diagnostic['MEPSPairs']} "
                f"MEPS_ER_AP={diagnostic['MEPS_ER_ValidationAP']:.4f} "
                f"MEPS_IP_AP={diagnostic['MEPS_IP_ValidationAP']:.4f}",
                flush=True,
            )
        tune_ids, tune_val_ids = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(outer_train, outer_train.target, groups=outer_train.patient_nbr))
        tune_train, tune_val = outer_train.iloc[tune_ids], outer_train.iloc[tune_val_ids]
        x_tune, x_val, _ = encode_frames(
            tune_train, tune_val, tune_val, columns, args.min_frequency,
        )
        _, hgb_params, hgb_iterations, _ = choose_hgb(
            x_tune, x_val, tune_train.target.to_numpy(), tune_val.target.to_numpy(),
            args.seed + fold,
        )

        print(f"[Fold {fold}] generating patient-disjoint HGB OOF risk", flush=True)
        base_oof = cross_fitted_hgb(
            outer_train, columns, hgb_params, hgb_iterations,
            args.correction_folds, args.seed + fold * 100, args.min_frequency,
        )
        selected_ap, selected_l2, _ = select_correction(
            outer_train, base_oof, proxy_columns, args.correction_l2,
            args.seed + fold * 1000,
        )

        raw_train = correction_features(
            base_oof, outer_train[proxy_columns[0]], outer_train[proxy_columns[1]],
        )
        correction_scaler = StandardScaler().fit(raw_train)
        if selected_l2 is None:
            correction_parameters = None
        else:
            correction_parameters = fit_offset_logistic(
                correction_scaler.transform(raw_train), outer_train.target.to_numpy(),
                clipped_logit(base_oof), selected_l2,
            )

        x_outer, _, x_test = encode_frames(
            outer_train, outer_train, test, columns, args.min_frequency,
        )
        final_hgb = hgb_model(
            args.seed + fold, {**hgb_params, "max_iter": hgb_iterations},
            early_stopping=False,
        ).fit(x_outer, outer_train.target.to_numpy())
        base_test = final_hgb.predict_proba(x_test)[:, 1]
        if correction_parameters is None:
            corrected_test = base_test.copy()
        else:
            raw_test = correction_features(
                base_test, test[proxy_columns[0]], test[proxy_columns[1]],
            )
            corrected_test = predict_offset_logistic(
                correction_scaler.transform(raw_test), clipped_logit(base_test),
                correction_parameters,
            )

        fold_oof = test[["encounter_id", "patient_nbr", "target"]].rename(
            columns={"target": "y_true"},
        ).copy()
        fold_oof["Fold"] = fold
        fold_oof["p_hgb_tuned"] = base_test
        fold_oof["p_meps_residual"] = corrected_test
        oof_parts.append(fold_oof)
        for name, probability in {
            "hgb_tuned": base_test, "meps_residual": corrected_test,
        }.items():
            rows.append({
                "Fold": fold, "Experiment": name,
                "PR_AUC_AP": average_precision_score(test.target, probability),
                "AUROC": roc_auc_score(test.target, probability),
                "Brier": brier_score_loss(test.target, probability),
            })
        selections.append({
            "Fold": fold, "CorrectionL2": selected_l2,
            "CorrectionFallback": selected_l2 is None,
            "CorrectionValidationAP": selected_ap,
            "HGBIterations": hgb_iterations,
            "HGBParameters": json.dumps(hgb_params, sort_keys=True),
        })
        print(
            f"[Fold {fold}] HGB={rows[-2]['PR_AUC_AP']:.6f} "
            f"MEPS_residual={rows[-1]['PR_AUC_AP']:.6f} "
            f"selected_L2={selected_l2 if selected_l2 is not None else 'fallback'}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(output / "partial_results.csv", index=False)

    result = pd.DataFrame(rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof, reference_models = attach_reference(oof, args.reference_oof)
    all_models = MODELS + reference_models
    if reference_models:
        for fold, part in oof.groupby("Fold", sort=True):
            probability = part.p_ensemble_fixed_50.to_numpy()
            rows.append({
                "Fold": int(fold), "Experiment": "ensemble_fixed_50",
                "PR_AUC_AP": average_precision_score(part.y_true, probability),
                "AUROC": roc_auc_score(part.y_true, probability),
                "Brier": brier_score_loss(part.y_true, probability),
            })
        result = pd.DataFrame(rows)

    summary = result.groupby("Experiment").agg(
        PR_AUC_mean=("PR_AUC_AP", "mean"), PR_AUC_std=("PR_AUC_AP", "std"),
        AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"),
    ).reset_index().sort_values("PR_AUC_mean", ascending=False)
    comparisons = COMPARISONS.copy()
    if reference_models:
        comparisons += [
            ("meps_residual", "ensemble_fixed_50"),
            ("ensemble_fixed_50", "hgb_tuned"),
        ]
    bootstrap = cluster_bootstrap(
        oof, all_models, comparisons, args.bootstrap, args.seed + 17000,
    )

    result.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    oof.to_csv(output / "oof_all.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "selected_correction.csv", index=False)
    if external_diagnostics:
        pd.DataFrame(external_diagnostics).to_csv(
            output / "domain_weighting_diagnostics.csv", index=False,
        )
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    config = vars(args).copy()
    config["reference_oof"] = str(args.reference_oof) if args.reference_oof else None
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nFold-stratified patient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default=MEPS_BRIDGE)
    parser.add_argument("--outdir", default=result_dir("meps_residual_correction"))
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument(
        "--external-method", choices=["harmonized_weighted", "legacy"],
        default="harmonized_weighted",
    )
    parser.add_argument("--domain-weight-min", type=float, default=.1)
    parser.add_argument("--domain-weight-max", type=float, default=10.)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--correction-folds", type=int, default=3)
    parser.add_argument(
        "--correction-l2", nargs="+", type=float,
        default=[.001, .01, .1, 1., 10.],
    )
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if (
        args.folds < 2 or args.correction_folds < 2 or args.bootstrap < 1
        or any(value < 0 for value in args.correction_l2)
        or args.domain_weight_min <= 0
        or args.domain_weight_max < args.domain_weight_min
    ):
        parser.error(
            "fold counts must be >= 2, bootstrap >= 1, L2 values >= 0, and "
            "domain weight bounds must satisfy 0 < min <= max"
        )
    run(args)


if __name__ == "__main__":
    main()
