#!/usr/bin/env python3
"""Align controlled diagnosis OOF, calculate metrics and patient bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
try:
    from numba import njit
except ImportError:
    def njit(*_args, **_kwargs):
        return lambda function: function

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diabetes_project.paper_comparison import top_fraction_metrics  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
CONFIG = {
    "LR": ("lr_standard", "lr_standard", "lr_standard"),
    "RF": ("rf_standard", "rf_standard", "rf_standard"),
    "XGBoost": ("xgb_standard", "xgb_standard", "xgb_standard"),
    "HGB": ("hgb_raw", "hgb_ccs_replace", "hgb_raw_plus_ccs"),
    "CatBoost": ("catboost_raw", "catboost_ccs_replace_diag", "catboost_raw_plus_ccs"),
}
VARIANTS = ("Raw", "CCS Replacement", "Raw + CCS")


def reference(path):
    frame = pd.read_csv(path, usecols=KEYS)
    if len(frame) != 99343 or frame.encounter_id.duplicated().any():
        raise ValueError("Reference cohort must contain 99,343 unique encounters")
    if frame.patient_nbr.nunique() != 69990 or set(frame.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Unexpected patient count or fold assignment")
    if frame.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Patient crosses outer folds")
    return frame.sort_values("encounter_id").reset_index(drop=True)


def align(ref, source, probability_col, name):
    if source.encounter_id.duplicated().any() or len(source) != len(ref):
        raise ValueError(f"{name}: duplicate or incomplete OOF")
    merged = ref.merge(source[KEYS + [probability_col]], on="encounter_id",
                       how="left", validate="one_to_one", suffixes=("", "_source"))
    if merged[probability_col].isna().any():
        raise ValueError(f"{name}: missing predictions")
    for key in KEYS[1:]:
        if not np.array_equal(merged[key].to_numpy(), merged[f"{key}_source"].to_numpy()):
            raise ValueError(f"{name}: mismatched {key}")
    probability = merged[probability_col].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"{name}: invalid probabilities")
    return probability


def long_probability(ref, path, model):
    source = pd.read_csv(path)
    source = source.loc[source.Model.eq(model), KEYS + ["probability"]]
    return align(ref, source, "probability", f"{path}:{model}")


def load_predictions(args, ref):
    output = {}
    raw = pd.read_csv(args.classic_raw, usecols=KEYS + ["Model", "probability"])
    replace = pd.read_csv(args.classic_replace, usecols=KEYS + ["Model", "probability"])
    append = pd.read_csv(args.classic_append, usecols=KEYS + ["Model", "probability"])
    hgb = pd.read_csv(args.hgb, usecols=KEYS + ["Model", "probability"])
    catboost = pd.read_csv(args.catboost, usecols=KEYS + list(CONFIG["CatBoost"]))
    for model, names in CONFIG.items():
        for variant, source_name in zip(VARIANTS, names):
            name = f"{model} | {variant}"
            if model in ("LR", "RF", "XGBoost"):
                table = {"Raw": raw, "CCS Replacement": replace, "Raw + CCS": append}[variant]
                selected = table.loc[table.Model.eq(source_name), KEYS + ["probability"]]
                output[name] = align(ref, selected, "probability", name)
            elif model == "HGB":
                selected = hgb.loc[hgb.Model.eq(source_name), KEYS + ["probability"]]
                output[name] = align(ref, selected, "probability", name)
            else:
                output[name] = align(ref, catboost, source_name, name)
    return output


def fold_tables(ref, predictions):
    rows = []
    for name, scores in predictions.items():
        model, variant = name.split(" | ")
        for fold in range(1, 6):
            select = ref.Fold.eq(fold).to_numpy()
            metric = top_fraction_metrics(ref.y_true.to_numpy()[select], scores[select])
            rows.append({"Model": model, "Representation": variant, "Fold": fold, **metric})
    fold = pd.DataFrame(rows)
    summary = fold.groupby(["Model", "Representation"], as_index=False).agg(
        Folds=("Fold", "nunique"), PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"), AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"), Precision_mean=("Precision", "mean"),
        Precision_std=("Precision", "std"), Recall_mean=("Recall", "mean"),
        Recall_std=("Recall", "std"), F1_mean=("F1_score", "mean"),
        F1_std=("F1_score", "std"), Brier_mean=("Brier_Score", "mean"),
        Brier_std=("Brier_Score", "std"),
    )
    y = ref.y_true.to_numpy()
    summary["Pooled_OOF_PR_AUC"] = [average_precision_score(y, predictions[f"{row.Model} | {row.Representation}"])
                                     for row in summary.itertuples()]
    order = pd.MultiIndex.from_product([list(CONFIG), VARIANTS], names=["Model", "Representation"])
    summary = summary.set_index(["Model", "Representation"]).loc[order].reset_index()
    return fold, summary


def comparisons(fold):
    lookup = fold.set_index(["Model", "Representation", "Fold"]).sort_index()
    rows = []
    for model in CONFIG:
        for treatment, control in (("CCS Replacement", "Raw"), ("Raw + CCS", "Raw"),
                                   ("Raw + CCS", "CCS Replacement")):
            a = lookup.loc[(model, treatment)]
            b = lookup.loc[(model, control)]
            rows.append({"Model": model, "Treatment": treatment, "Control": control,
                         "Delta_PR_AUC": float((a.PR_AUC - b.PR_AUC).mean()),
                         "Delta_AUROC": float((a.AUROC - b.AUROC).mean()),
                         "Delta_F1": float((a.F1_score - b.F1_score).mean()),
                         "Delta_Brier": float((a.Brier_Score - b.Brier_Score).mean())})
    return pd.DataFrame(rows)


@njit(cache=True)
def _sorted_metric(y_sorted, score_sorted, squared_error, sorted_weights):
    """Weighted AP, AUROC, top-10%-risk F1 and Brier; ties share one threshold."""
    n = len(y_sorted)
    total = 0
    positives = 0
    brier_sum = 0.0
    for i in range(n):
        total += sorted_weights[i]
        positives += sorted_weights[i] * y_sorted[i]
        brier_sum += sorted_weights[i] * squared_error[i]
    negatives = total - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("A patient bootstrap draw contains only one class")
    alerted = max(1, int(np.floor(0.1 * total)))
    seen = 0
    cumulative_positive = 0
    cumulative_negative = 0
    group_positive = 0
    group_negative = 0
    alerted_tp = 0
    ap_sum = 0.0
    auc_sum = 0.0
    for i in range(n):
        weight = sorted_weights[i]
        if seen < alerted:
            taken = min(weight, alerted - seen)
            alerted_tp += taken * y_sorted[i]
        seen += weight
        group_positive += weight * y_sorted[i]
        group_negative += weight * (1 - y_sorted[i])
        if i == n - 1 or score_sorted[i] != score_sorted[i + 1]:
            cumulative_positive += group_positive
            cumulative_negative += group_negative
            group_total = cumulative_positive + cumulative_negative
            if group_total > 0:
                ap_sum += (cumulative_positive / group_total) * group_positive
            auc_sum += group_positive * (negatives - cumulative_negative + 0.5 * group_negative)
            group_positive = 0
            group_negative = 0
    ap = ap_sum / positives
    auroc = auc_sum / (positives * negatives)
    f1 = 2.0 * alerted_tp / (alerted + positives)
    brier = brier_sum / total
    return np.array([ap, auroc, f1, brier])


def bootstrap_metrics(ref, predictions, comparison, reps, seed):
    """One patient draw per fold/repeat, shared by all fifteen predictions."""
    rng = np.random.default_rng(seed + 17000)
    names = list(predictions)
    index = {name: i for i, name in enumerate(names)}
    folds = []
    y = ref.y_true.to_numpy(dtype=int)
    for fold in range(1, 6):
        selected = np.flatnonzero(ref.Fold.to_numpy() == fold)
        _, inverse = np.unique(ref.patient_nbr.to_numpy()[selected], return_inverse=True)
        prepared = []
        for name in names:
            scores = predictions[name][selected]
            order = np.argsort(-scores, kind="mergesort")
            sorted_scores = scores[order]
            prepared.append((order, y[selected][order], sorted_scores,
                             (sorted_scores - y[selected][order]) ** 2))
        folds.append((inverse, int(inverse.max()) + 1, prepared))
    draws = np.empty((reps, len(names), 4), dtype=float)
    for repeat in range(reps):
        fold_values = np.empty((5, len(names), 4), dtype=float)
        for fold_index, (inverse, patients, prepared) in enumerate(folds):
            counts = np.bincount(rng.integers(patients, size=patients), minlength=patients)
            weights = counts[inverse]
            for model_index, (order, sorted_y, sorted_scores, error) in enumerate(prepared):
                fold_values[fold_index, model_index] = _sorted_metric(
                    sorted_y, sorted_scores, error, weights[order],
                )
        draws[repeat] = fold_values.mean(axis=0)
        if (repeat + 1) % 250 == 0:
            print(f"[Bootstrap] {repeat + 1}/{reps}", flush=True)
    rows = []
    for row in comparison.itertuples():
        a = index[f"{row.Model} | {row.Treatment}"]
        b = index[f"{row.Model} | {row.Control}"]
        delta = draws[:, a] - draws[:, b]
        record = {"Model": row.Model, "Treatment": row.Treatment, "Control": row.Control,
                  "BootstrapReplicates": reps, "ResamplingUnit": "patient within fold"}
        for metric, point, samples in zip(
            ("PR_AUC", "AUROC", "F1", "Brier"),
            (row.Delta_PR_AUC, row.Delta_AUROC, row.Delta_F1, row.Delta_Brier), delta.T,
        ):
            low, high = np.quantile(samples, [0.025, 0.975])
            record[f"Delta_{metric}"] = point
            record[f"{metric}_CI95_Lower"] = low
            record[f"{metric}_CI95_Upper"] = high
            record[f"{metric}_FractionPositive"] = float(np.mean(samples > 0))
        rows.append(record)
    return pd.DataFrame(rows)


def markdown_table(frame, names):
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def reports(args, fold, summary, comparison, bootstrap, feature_counts):
    columns = ["Model", "Representation", "PR_AUC_mean", "AUROC_mean", "Precision_mean",
               "Recall_mean", "F1_mean", "Brier_mean"]
    labels = ["Model", "Representation", "PR-AUC (AP)", "AUROC", "Precision", "Recall", "F1", "Brier"]
    main = summary[summary.Model.isin(["XGBoost", "HGB", "CatBoost"])][columns]
    ap = summary.pivot(index="Model", columns="Representation", values="PR_AUC_mean").loc[
        ["XGBoost", "HGB", "CatBoost"]]
    compact = pd.DataFrame({"Model": ap.index, "Raw AP": ap["Raw"].to_numpy(),
                            "CCS AP": ap["CCS Replacement"].to_numpy(),
                            "Raw + CCS AP": ap["Raw + CCS"].to_numpy(),
                            "Delta AP vs Raw": (ap["Raw + CCS"] - ap["Raw"]).to_numpy()})
    (args.outdir / "main_experiment_table.md").write_text(
        "# Controlled diagnosis representations: main slide\n\n" + markdown_table(main, labels)
        + "\n\n## Compact AP comparison\n\n" + markdown_table(compact, list(compact)) + "\n", encoding="utf-8")
    appendix = summary[columns + ["PR_AUC_std", "AUROC_std", "Precision_std", "Recall_std",
                                "F1_std", "Brier_std", "Pooled_OOF_PR_AUC"]]
    fold_view = fold[["Model", "Representation", "Fold", "PR_AUC", "AUROC", "Precision",
                      "Recall", "F1_score", "Brier_Score"]]
    bootstrap_view = bootstrap[["Model", "Treatment", "Control", "Delta_PR_AUC",
                                "PR_AUC_CI95_Lower", "PR_AUC_CI95_Upper", "Delta_AUROC",
                                "AUROC_CI95_Lower", "AUROC_CI95_Upper", "Delta_F1",
                                "F1_CI95_Lower", "F1_CI95_Upper", "Delta_Brier",
                                "Brier_CI95_Lower", "Brier_CI95_Upper"]]
    (args.outdir / "appendix_full_table.md").write_text(
        "# Appendix: all models and diagnosis representations\n\n" + markdown_table(appendix, list(appendix))
        + "\n\n## Fold-wise results\n\n" + markdown_table(fold_view, list(fold_view))
        + "\n\n## Paired patient bootstrap (2,000 repetitions)\n\n"
        + markdown_table(bootstrap_view, list(bootstrap_view))
        + "\n\n## Feature counts\n\n" + markdown_table(feature_counts, list(feature_counts))
        + "\n\nCatBoost uses model-native categorical handling; its encoded dimension is not a fixed one-hot count.\n"
        + "\n## Controlled training settings\n\n"
        + "- LR/RF/XGBoost: `run_xgb_5fold_recheck.py`; train-fold numeric median imputation and scaling, "
          "categorical mode imputation and one-hot encoding. LR: C=1, lbfgs, 1500 iterations. "
          "RF: 300 trees, max depth 16, minimum leaf 2. XGBoost: depth 4, learning rate 0.05, "
          "500-tree limit, 30-round early stopping on a patient-disjoint 20% inner validation. "
          "All three representations were rerun on CPU with eight threads and no class weighting.\n"
        + "- HGB: `run_diagnosis_representation_hgb.py`; train-fold median/constant imputation, "
          "one-hot encoding with minimum frequency 100. Five candidates from `run_nested_hgb_blend.py` "
          "are selected by inner-validation AP, then refit on the full outer train.\n"
        + "- CatBoost: saved OOF from `run_ccs_catboost_5fold.py` and "
          "`run_paper_fair_comparison.py`; raw diagnosis codes remain categorical. "
          "Depths 4/6 and 250/500/750 trees are chosen by inner-validation AP, "
          "then refit on the full outer train. CatBoost random seed is 2026 for all three variants.\n"
        + "- CCS mapping: `load_ccs` and `attach_uci_ccs` from the existing project; "
          "the three representations have 46, 46 and 49 input columns respectively.\n"
        + "- Every model uses the saved seed-777 patient-disjoint outer folds. "
          "Precision/Recall/F1 use the highest 10% test risks in each fold. "
          "No external data, class weight, SMOTE or SMOTENC are used.\n",
        encoding="utf-8")
    findings = []
    for model in ("XGBoost", "HGB", "CatBoost"):
        part = summary[summary.Model.eq(model)].set_index("Representation")
        best = part.PR_AUC_mean.idxmax()
        delta = part.loc["Raw + CCS", "PR_AUC_mean"] - part.loc["Raw", "PR_AUC_mean"]
        ci = bootstrap[(bootstrap.Model.eq(model)) & bootstrap.Treatment.eq("Raw + CCS") & bootstrap.Control.eq("Raw")].iloc[0]
        findings.append(f"{model}: {best} had the highest five-fold mean AP ({part.loc[best, 'PR_AUC_mean']:.4f}); "
                        f"adding CCS to raw ICD-9 changed AP by {delta:+.4f} "
                        f"(patient bootstrap 95% CI [{ci.PR_AUC_CI95_Lower:+.4f}, {ci.PR_AUC_CI95_Upper:+.4f}]).")
    existing = pd.read_csv(args.catboost.parent / "summary.csv").set_index("Model")
    previous = pd.read_csv(args.catboost.parent / "paired_ap_bootstrap.csv")
    ensemble = previous.loc[
        previous.Treatment.eq("ccs_hgb_catboost_raw_plus_ccs_50")
        & previous.Control.eq("ccs_hgb_catboost_raw_50")
    ].iloc[0]
    (args.outdir / "experiment_findings.md").write_text(
        "# Key findings for presentation\n\n" + "\n".join(f"- {line}" for line in findings)
        + "\n\n## Existing ensemble decision\n\nThe original CCS HGB + raw CatBoost ensemble has five-fold mean "
        f"AP {existing.loc['ccs_hgb_catboost_raw_50', 'PR_AUC_mean']:.6f}; using raw+CCS CatBoost "
        f"gives {existing.loc['ccs_hgb_catboost_raw_plus_ccs_50', 'PR_AUC_mean']:.6f}. "
        f"The paired AP difference is {ensemble.MeanFold_Delta_PR_AUC:+.6f} "
        f"(95% CI [{ensemble.CI95_Lower:+.6f}, {ensemble.CI95_Upper:+.6f}]). "
        "This comparison does not establish an ensemble improvement.\n\n"
        "## Interpretation\n\nThis experiment evaluates diagnosis representation within each model family. "
        "The earlier GPU classic baselines are historical references; the controlled classic comparison "
        "uses fresh CPU runs for all three representations. The saved CatBoost variants are reused "
        "because their feature construction, selection grid and outer folds match.\n",
        encoding="utf-8")


def compute_feature_counts(args):
    """Re-fit only training-fold encoders to report actual encoded dimensions."""
    from diabetes_project.features import make_preprocessor
    from run_xgb_5fold_recheck import read_split
    from run_meps_hgb import normalize, raw_discharge_predictors
    from run_meps_ccs_group_hgb import DIAG, attach_uci_ccs
    from train_xgboost_ahrq_ccs_prefix import load_ccs

    cohort, folds = read_split(ROOT / "data/raw/uci/diabetic_data.csv", args.reference_oof)
    raw_columns = raw_discharge_predictors(cohort)
    mapping, _ = load_ccs(ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    cohort = attach_uci_ccs(cohort, mapping)
    variants = {
        "Raw": raw_columns,
        "CCS Replacement": [f"ccs_{column}" if column in DIAG else column for column in raw_columns],
        "Raw + CCS": raw_columns + [f"ccs_{column}" for column in DIAG],
    }
    rows = []
    for variant, columns in variants.items():
        frame, categorical, _ = normalize(cohort, columns)
        for column in categorical:
            frame[column] = frame[column].astype(object)
        for fold in range(1, 6):
            train = frame.iloc[np.flatnonzero(folds != fold)]
            prep = make_preprocessor(train, columns).fit(train)
            encoded = len(prep.get_feature_names_out())
            for model in ("LR", "RF", "XGBoost"):
                rows.append({"Model": model, "Representation": variant, "Fold": fold,
                             "Input_Features": len(columns), "Encoded_Features": encoded})
            rows.append({"Model": "CatBoost", "Representation": variant, "Fold": fold,
                         "Input_Features": len(columns), "Encoded_Features": "model-native; not fixed"})
    hgb = pd.read_csv(args.hgb_counts)
    variant_names = {"hgb_raw": "Raw", "hgb_ccs_replace": "CCS Replacement",
                     "hgb_raw_plus_ccs": "Raw + CCS"}
    for row in hgb.itertuples():
        rows.append({"Model": "HGB", "Representation": variant_names[row.Model],
                     "Fold": row.Fold, "Input_Features": row.Input_Features,
                     "Encoded_Features": row.Outer_Encoded_Features})
    return pd.DataFrame(rows).sort_values(["Model", "Representation", "Fold"])


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    ref = reference(args.reference_oof)
    predictions = load_predictions(args, ref)
    fold, summary = fold_tables(ref, predictions)
    comparison = comparisons(fold)
    fold.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary.to_csv(args.outdir / "raw_ccs_5fold_summary.csv", index=False)
    comparison.to_csv(args.outdir / "diagnosis_representation_comparison.csv", index=False)
    oof = ref.copy()
    for name, scores in predictions.items():
        oof[name.replace(" | ", "__").replace(" + ", "_plus_").replace(" ", "_")] = scores
    oof.to_csv(args.outdir / "oof_all.csv.gz", index=False)
    bootstrap = bootstrap_metrics(ref, predictions, comparison, args.bootstrap, args.seed)
    bootstrap.to_csv(args.outdir / "diagnosis_representation_bootstrap.csv", index=False)
    feature_counts = compute_feature_counts(args)
    feature_counts.to_csv(args.outdir / "feature_counts.csv", index=False)
    reports(args, fold, summary, comparison, bootstrap, feature_counts)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": args.seed, "folds": 5, "cohort_rows": len(ref),
        "patients": int(ref.patient_nbr.nunique()), "reference_oof": str(args.reference_oof.resolve()),
        "sources": {key: str(getattr(args, key).resolve()) for key in
                    ("classic_raw", "classic_replace", "classic_append", "hgb", "catboost")},
        "threshold_policy": "top 10% highest risk per outer fold",
        "bootstrap": f"patient resampling within fold, paired AP/AUROC/top-10%-F1/Brier, {args.bootstrap} replicates",
        "metric": "sklearn average_precision_score; arithmetic mean over five folds",
        "reuse": "CatBoost raw, replacement, append",
        "new_training": "LR/RF/XGB raw, replacement, append on CPU; all three HGB variants",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.round(6).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-oof", type=Path,
                        default=ROOT / "results/meps_ccs_group_hgb_seed777/oof_all.csv")
    parser.add_argument("--classic-raw", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/classic_raw/oof_long.csv")
    parser.add_argument("--classic-replace", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/classic_replace/oof_long.csv")
    parser.add_argument("--classic-append", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/classic_append/oof_long.csv")
    parser.add_argument("--hgb", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/hgb_controlled/oof_all.csv.gz")
    parser.add_argument("--hgb-counts", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold/hgb_controlled/feature_counts.csv")
    parser.add_argument("--catboost", type=Path,
                        default=ROOT / "results/ccs_catboost_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "results/diagnosis_raw_ccs_5fold")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.seed != 777 or args.bootstrap < 1:
        parser.error("Use the saved seed 777 folds and a positive bootstrap count")
    run(args)


if __name__ == "__main__":
    main()
