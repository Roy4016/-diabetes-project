#!/usr/bin/env python3
"""Audit saved five-fold OOF probabilities and evaluate their calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
MODELS = {
    "catboost_raw": "Raw ICD-9 CatBoost",
    "ccs_only_hgb_catboost_raw_50": "CCS HGB + Raw CatBoost (50/50)",
    "cms_meps_cdc_hgb50": "CDC + MEPS + CMS CatBoost + CCS HGB (50/50)",
}
ORIGINAL = "ccs_only_hgb_catboost_raw_50"
EXTERNAL = "cms_meps_cdc_hgb50"
EXPECTED_BRIER = {
    "catboost_raw": 0.09508422865574859,
    ORIGINAL: 0.09506791013298885,
    EXTERNAL: 0.09507342607348516,
}


def load_oof(path: Path, paper_path: Path) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(path, usecols=KEYS + list(MODELS))
    if len(data) != 99_343 or data.encounter_id.isna().any() or data.encounter_id.duplicated().any():
        raise ValueError("Expected 99,343 unique, nonmissing UCI encounter IDs")
    if data[KEYS].isna().any().any() or set(data.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Missing key or incorrect five-fold assignments")
    if data.patient_nbr.nunique() != 69_990 or data.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Patients are not disjoint across the original five folds")
    if not data.y_true.isin([0, 1]).all():
        raise ValueError("Readmission labels must be binary")
    for model in MODELS:
        probability = data[model].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid or missing OOF probability: {model}")

    saved = pd.read_csv(paper_path, usecols=KEYS + ["catboost_raw", "ccs_only_catboost_raw_50"])
    if len(saved) != len(data) or saved.encounter_id.duplicated().any():
        raise ValueError("Saved comparison has different encounter coverage")
    aligned = saved.set_index("encounter_id").loc[data.encounter_id].reset_index()
    if not aligned[KEYS].equals(data[KEYS]):
        raise ValueError("Patient, label or fold mismatch against paper comparison OOF")
    for new, old in (("catboost_raw", "catboost_raw"),
                     (ORIGINAL, "ccs_only_catboost_raw_50")):
        if not np.allclose(data[new], aligned[old], rtol=0, atol=1e-12):
            raise ValueError(f"Baseline probabilities differ against paper OOF: {new}")
    return data, {
        "rows": len(data), "patients": int(data.patient_nbr.nunique()),
        "positives": int(data.y_true.sum()), "positive_rate": float(data.y_true.mean()),
        "fold_rows": {str(k): int(v) for k, v in data.Fold.value_counts().sort_index().items()},
        "paper_baselines_match": True, "patient_disjoint": True,
    }


def quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    return np.percentile(reference, np.linspace(0, 100, bins + 1))


def bin_statistics(y: np.ndarray, probability: np.ndarray, edges: np.ndarray,
                   scope: str, fold: int | str, model: str, policy: str) -> pd.DataFrame:
    assignment = np.searchsorted(edges[1:-1], probability, side="left")
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = assignment == bin_id
        if not mask.any():
            continue
        actual = float(y[mask].mean())
        predicted = float(probability[mask].mean())
        rows.append({
            "Scope": scope, "Fold": fold, "Model": model, "Binning": policy,
            "Bin": bin_id + 1, "Lower_Edge": float(edges[bin_id]),
            "Upper_Edge": float(edges[bin_id + 1]), "N": int(mask.sum()),
            "Mean_Predicted": predicted, "Observed_Rate": actual,
            "Signed_Gap_PredMinusObserved": predicted - actual,
            "Absolute_Calibration_Error": abs(predicted - actual),
        })
    return pd.DataFrame(rows)


def sklearn_quantile_statistics(y: np.ndarray, probability: np.ndarray, bins: int,
                                scope: str, fold: int | str, model: str) -> pd.DataFrame:
    observed, predicted = calibration_curve(y, probability, n_bins=bins, strategy="quantile")
    table = bin_statistics(y, probability, quantile_edges(probability, bins),
                           scope, fold, model, "model_quantile_sklearn")
    if len(table) != len(observed) or not np.allclose(table.Observed_Rate, observed, atol=1e-12) \
            or not np.allclose(table.Mean_Predicted, predicted, atol=1e-12):
        raise ValueError(f"Bin calculation disagrees with sklearn.calibration_curve: {model}, {fold}")
    return table


def ece(table: pd.DataFrame) -> float:
    return float(np.average(table.Absolute_Calibration_Error, weights=table.N))


def bootstrap_fold_arrays(part: pd.DataFrame, edges: np.ndarray) -> dict:
    patients, patient_index = np.unique(part.patient_nbr.to_numpy(), return_inverse=True)
    n_patients, n_bins = len(patients), len(edges) - 1
    y = part.y_true.to_numpy(dtype=float)
    arrays = {"n_patients": n_patients, "n_rows": len(part)}
    for model in (ORIGINAL, EXTERNAL):
        p = part[model].to_numpy(dtype=float)
        bin_index = np.searchsorted(edges[1:-1], p, side="left")
        residual = np.zeros((n_patients, n_bins), dtype=float)
        np.add.at(residual, (patient_index, bin_index), y - p)
        brier_sum = np.bincount(patient_index, weights=(y - p) ** 2,
                                minlength=n_patients)
        arrays[model] = (residual, brier_sum)
    return arrays


def bootstrap(data: pd.DataFrame, fold_edges: dict[int, np.ndarray],
              replicates: int, seed: int) -> pd.DataFrame:
    fold_arrays = [bootstrap_fold_arrays(part, fold_edges[int(fold)])
                   for fold, part in data.groupby("Fold", sort=True)]

    def estimates(arrays: dict, counts: np.ndarray) -> tuple[float, float]:
        total = float(np.dot(counts, np.ones_like(counts)) if counts is not None
                      else arrays["n_rows"])
        values = []
        for model in (ORIGINAL, EXTERNAL):
            residual, brier_sum = arrays[model]
            weighted_residual = residual.sum(axis=0) if counts is None else counts @ residual
            weighted_brier = brier_sum.sum() if counts is None else counts @ brier_sum
            values.append((float(weighted_brier / total),
                           float(np.abs(weighted_residual).sum() / total)))
        return values[1][0] - values[0][0], values[1][1] - values[0][1]

    observed = np.mean([estimates(a, None) for a in fold_arrays], axis=0)
    rng = np.random.default_rng(seed + 17_000)
    samples = np.empty((replicates, 2), dtype=float)
    for replicate in range(replicates):
        differences = []
        for arrays in fold_arrays:
            size = arrays["n_patients"]
            counts = np.bincount(rng.integers(0, size, size), minlength=size)
            differences.append(estimates(arrays, counts))
        samples[replicate] = np.mean(differences, axis=0)
    records = []
    for index, name in enumerate(("Brier", "ECE")):
        low, high = np.quantile(samples[:, index], [0.025, 0.975])
        records.append({
            "Comparison": "External Ensemble - Original Ensemble", "Metric": name,
            "MeanFold_Delta": observed[index], "CI95_Lower": low, "CI95_Upper": high,
            "FractionNegative": float(np.mean(samples[:, index] < 0)),
            "BootstrapReplicates": replicates,
            "ResamplingUnit": "patient within fold",
            "Binning": "fixed fold-specific original-ensemble quantile edges",
        })
    return pd.DataFrame(records)


def plot_pooled(bins: pd.DataFrame, target: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 6))
    axis.plot([0, 1], [0, 1], "--", color="0.4", label="Perfect calibration")
    for name, label in MODELS.items():
        part = bins[(bins.Scope == "pooled_oof") & (bins.Binning == "model_quantile_sklearn")
                    & (bins.Model == name)].sort_values("Bin")
        axis.plot(part.Mean_Predicted, part.Observed_Rate, marker="o", linewidth=2,
                  markersize=5, label=label)
    axis.set(xlabel="Mean predicted readmission probability",
             ylabel="Observed 30-day readmission frequency", title="Pooled OOF calibration")
    axis.set_xlim(0, 0.5)
    axis.set_ylim(0, 0.5)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def plot_fivefold(bins: pd.DataFrame, target: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for fold, axis in enumerate(axes.flat, start=1):
        if fold > 5:
            axis.axis("off")
            break
        axis.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1)
        for name, label in MODELS.items():
            part = bins[(bins.Scope == "fold") & (bins.Fold == fold)
                        & (bins.Binning == "model_quantile_sklearn")
                        & (bins.Model == name)].sort_values("Bin")
            axis.plot(part.Mean_Predicted, part.Observed_Rate, marker="o",
                      markersize=3, linewidth=1.5, label=label)
        axis.set_title(f"Outer fold {fold}")
        axis.set_xlim(0, 0.5)
        axis.set_ylim(0, 0.5)
        axis.grid(alpha=0.2)
    fig.supxlabel("Mean predicted readmission probability")
    fig.supylabel("Observed 30-day readmission frequency")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.11), fontsize=9)
    fig.suptitle("Five-fold OOF calibration (10 quantile bins per model)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
    fig.savefig(target, dpi=180)
    plt.close(fig)


def write_summary(output: Path, audit: dict, metrics: pd.DataFrame,
                  boot: pd.DataFrame, bins: pd.DataFrame, args) -> None:
    means = metrics[metrics.Scope == "fivefold_mean"].set_index("Model")
    pooled = metrics[metrics.Scope == "pooled_oof"].set_index("Model")
    lines = [
        "# 五折校準與預測機率可靠度分析", "",
        "## 設定與 OOF 核對", "",
        f"UCI 共 **{audit['rows']:,}** 次就醫、**{audit['patients']:,}** 位病人；"
        f"正例 **{audit['positives']:,}** 筆（{audit['positive_rate']:.4%}）。"
        "使用 seed 777 的病人互斥五折 OOF 原始機率，沒有重新訓練或校準模型。"
        "Encounter ID、Patient ID、Fold、Label 均完整；原始 CatBoost 與 CCS HGB＋Raw CatBoost"
        "機率已逐筆和論文比較 OOF 核對。", "",
        "10 箱 reliability diagram 使用 `sklearn.calibration.calibration_curve(strategy='quantile')`，"
        "每個模型按自己的機率分位數分箱。定量 ECE 和配對 bootstrap 使用**各折原融合模型**"
        "的 10 個分位箱邊界，並將同一組邊界套到三個模型。"
        "因此圖中的分箱與 ECE 分箱不同；`calibration_bin_statistics.csv` 用 `Binning` 明列兩者。"
        "ECE = Σ_b (n_b/N)|observed_b − predicted_b|；空箱不補值。", "",
        "## 三模型比較", "",
        "| 模型 | 五折平均 Brier ↓ | 五折 Brier SD | Pooled Brier ↓ | 五折平均 ECE ↓ | 五折 ECE SD | Pooled ECE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in MODELS.items():
        m, p = means.loc[name], pooled.loc[name]
        lines.append(f"| {label} | {m.Brier:.6f} | {m.Brier_SD:.6f} | {p.Brier:.6f} | "
                     f"{m.ECE:.6f} | {m.ECE_SD:.6f} | {p.ECE:.6f} |")
    lines.extend(["", "五折平均與 pooled OOF 是不同的估計方式，已分開報告。", "",
                  "## 配對病人 bootstrap", "",
                  "比較方向固定為 **外部資料融合 − 原融合模型**；Brier、ECE 差值小於零表示較低。"
                  "每折以病人為抽樣單位，兩模型使用相同抽樣；先算各折差值再平均。"
                  "ECE 的箱界固定為原始完整 OOF 的各折分位數，bootstrap 只重新計算箱內人數、"
                  "正例率與平均機率，**不重新估計箱界**。", "",
                  "| 指標 | 平均五折差值 | 95% CI |",
                  "|---|---:|---:|"])
    for row in boot.itertuples():
        lines.append(f"| {row.Metric} | {row.MeanFold_Delta:+.6f} | "
                     f"[{row.CI95_Lower:+.6f}, {row.CI95_Upper:+.6f}] |")
    lines.extend(["", "CI 跨零不能證明兩模型等效。Brier 同時受排序、機率誤差及結果不確定性影響；"
                  "ECE 依分箱策略而變，不能單獨證明機率可靠。", "",
                  "## 曲線解讀與限制", ""])
    for name, label in MODELS.items():
        p = pooled.loc[name]
        direction = "整體偏高估" if p.Mean_Predicted > p.Observed_Rate else "整體偏低估"
        part = bins[(bins.Scope == "pooled_oof") & (bins.Binning == "model_quantile_sklearn")
                    & (bins.Model == name)]
        lines.append(f"- **{label}**：整體平均預測 {p.Mean_Predicted:.4f}、觀察率 "
                     f"{p.Observed_Rate:.4f}，屬於**{direction}**；10 箱最大絕對偏差 "
                     f"{part.Absolute_Calibration_Error.max():.4f}。仍須看不同風險箱是否交叉或局部偏離。")
    lines.extend(["", "圖：`calibration_curve_pooled.png`、`calibration_curve_fivefold.png`。"
                  "分位箱的模型各自邊界讓每箱樣本數接近，但曲線點不能視為共同風險區間。"
                  "定量比較使用共同箱界；其他模型在某些箱可能很少或沒有樣本。"
                  "本分析未檢驗族群或醫院間的校準，也未做外部前瞻驗證。", "",
                  "## English slide-ready conclusion", "",
                  f"- Across five patient-disjoint folds, the original ensemble had a mean Brier score "
                  f"of {means.loc[ORIGINAL, 'Brier']:.6f} and common-bin ECE of "
                  f"{means.loc[ORIGINAL, 'ECE']:.6f}.",
                  f"- The CDC + MEPS + CMS ensemble had a mean Brier score of "
                  f"{means.loc[EXTERNAL, 'Brier']:.6f} and common-bin ECE of "
                  f"{means.loc[EXTERNAL, 'ECE']:.6f}.",
                  "- Paired patient bootstrap quantifies the differences; these plots assess reliability "
                  "of existing probabilities, without recalibration.", "",
                  "## 重現指令", "", "```bash",
                  "cd /media/synology/G/314554044/diabetes_project",
                  "python scripts/experiments/analyze_calibration_5fold.py \\",
                  f"  --oof {args.oof.resolve().relative_to(ROOT)} \\",
                  f"  --paper-oof {args.paper_oof.resolve().relative_to(ROOT)} \\",
                  f"  --outdir {args.outdir.resolve().relative_to(ROOT)} \\",
                  f"  --bootstrap {args.bootstrap} --seed {args.seed}",
                  "```", ""])
    (output / "CALIBRATION_ANALYSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    if args.seed != 777 or args.bins != 10 or args.bootstrap < 1:
        raise ValueError("Use seed 777, 10 bins and at least one bootstrap replicate")
    data, audit = load_oof(args.oof, args.paper_oof)
    bin_tables, metric_rows, fold_edges = [], [], {}
    scopes = [("fold", int(fold), part) for fold, part in data.groupby("Fold", sort=True)]
    scopes.append(("pooled_oof", "all", data))
    for scope, fold, part in scopes:
        y = part.y_true.to_numpy(dtype=int)
        common_edges = quantile_edges(part[ORIGINAL].to_numpy(dtype=float), args.bins)
        if scope == "fold":
            fold_edges[int(fold)] = common_edges
        for name in MODELS:
            p = part[name].to_numpy(dtype=float)
            model_bins = sklearn_quantile_statistics(y, p, args.bins, scope, fold, name)
            shared_bins = bin_statistics(y, p, common_edges, scope, fold, name,
                                         "shared_original_quantile")
            bin_tables.extend((model_bins, shared_bins))
            metric_rows.append({
                "Scope": scope, "Fold": fold, "Model": name,
                "Brier": float(brier_score_loss(y, p)), "ECE": ece(shared_bins),
                "Mean_Predicted": float(p.mean()), "Observed_Rate": float(y.mean()),
                "N": len(part), "Binning": "shared_original_quantile",
            })
    bins = pd.concat(bin_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    fold_metrics = metrics[metrics.Scope == "fold"]
    summary = fold_metrics.groupby("Model", sort=False).agg(
        Brier=("Brier", "mean"), Brier_SD=("Brier", "std"),
        ECE=("ECE", "mean"), ECE_SD=("ECE", "std"),
        Mean_Predicted=("Mean_Predicted", "mean"), Observed_Rate=("Observed_Rate", "mean"),
        N=("N", "sum"),
    ).reset_index().assign(Scope="fivefold_mean", Fold="all",
                           Binning="shared_original_quantile")
    metrics["Brier_SD"] = np.nan
    metrics["ECE_SD"] = np.nan
    metrics = pd.concat([metrics, summary[metrics.columns]], ignore_index=True)
    for model, expected in EXPECTED_BRIER.items():
        actual = float(summary.set_index("Model").loc[model, "Brier"])
        if not np.isclose(actual, expected, atol=5e-7, rtol=0):
            raise ValueError(f"Brier mean mismatch for {model}: {actual:.9f} vs {expected:.9f}")
    print("[Audit] exact paper baselines match; five folds and 69,990 patients verified")
    print(metrics.loc[metrics.Scope == "fivefold_mean",
                      ["Model", "Brier", "Brier_SD", "ECE", "ECE_SD"]].to_string(index=False))
    boot = bootstrap(data, fold_edges, args.bootstrap, args.seed)
    print(boot[["Metric", "MeanFold_Delta", "CI95_Lower", "CI95_Upper"]].to_string(index=False))
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.outdir / "calibration_metrics.csv", index=False)
    bins.to_csv(args.outdir / "calibration_bin_statistics.csv", index=False)
    boot.to_csv(args.outdir / "calibration_paired_bootstrap.csv", index=False)
    plot_pooled(bins, args.outdir / "calibration_curve_pooled.png")
    plot_fivefold(bins, args.outdir / "calibration_curve_fivefold.png")
    write_summary(args.outdir, audit, metrics, boot, bins, args)
    (args.outdir / "oof_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved calibration analysis to {args.outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "results/cms_meps_cdc_catboost_seed777/oof_all.csv.gz")
    parser.add_argument("--paper-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/calibration_analysis_5fold_seed777")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
