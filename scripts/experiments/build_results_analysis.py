#!/usr/bin/env python3
"""Rebuild the five-fold-only RESULTS_ANALYSIS.md from saved results."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

SKIP = {
    "tabpfn35_meps_seed777", "meps_ccs_hgb_inner_cv_seed777",
    "paper_comparison_audit_seed777", "cdc_encoder_control_smoke",
}
PREFERRED = [
    "paper_fair_comparison_seed777", "classic_5fold_recheck_seed777",
    "meps_30d_joint_transfer_seed777", "meps_nhanes_dual_external_seed777",
    "cdc_clinical_interpretability_results", "cdc_diffwalk_control_results",
]
METRIC_COLS = ["AP", "AUROC", "Precision", "Recall", "F1", "Brier"]
DESCRIPTIONS = {
    "cdc_clinical_results": "CDC 健康與慢性病特徵加入醫院臨床模型，並與臨床基線比較。",
    "cdc_comorbidity_results": "將 CDC 共病相關資訊與醫院診斷特徵結合。",
    "cdc_diagnosis_aware_results": "以醫院診斷分組對齊 CDC 外部健康訊號。",
    "cdc_distribution_alignment_results": "對 CDC 與醫院資料的分布差異做對齊，再比較主要任務表現。",
    "cdc_encoder_control_results": "比較 CDC 編碼器、置換控制與僅使用醫院資料的模型。",
    "cdc_encoder_cv_results": "以五折驗證 CDC 編碼器轉移與臨床模型。",
    "cdc_ih_5fold_results": "比較 CDC 外部健康訊號在不同醫院特徵集合下的效果。",
    "cdc_clinical_interpretability_results": "CDC 健康指標加到醫院臨床特徵，另做單一健康欄位移除／保留控制與 SHAP。",
    "cdc_diffwalk_control_results": "對 CDC 行動困難 DiffWalk 做真實值、置換值及全部健康指標控制。",
    "meps_augmented_hgb_seed777": "把 MEPS 風險特徵直接加到醫院 HGB，對照既有 HGB 和固定融合。",
    "meps_bridge8_results": "以年齡、性別、疾病等 Bridge8 共享變數接 MEPS ER／住院利用率。",
    "meps_bridge8_seed2024": "Bridge8 方法以不同模型／分割 seed 檢查結果穩定性。",
    "meps_bridge8_seed3407": "Bridge8 方法以不同模型／分割 seed 檢查結果穩定性。",
    "meps_ccs_group_hgb_seed777": "把 MEPS 次年利用率依 AHRQ CCS 疾病群組彙整，再加到 HGB。",
    "meps_cell_results": "以人口與健康／利用率 cell 匹配 MEPS 的群組風險。",
    "meps_embedding_tuned_results": "調整 MEPS embedding 對醫院 XGBoost 的輸入方式。",
    "meps_embedding_xgb_results": "比較 MEPS 預訓練、多任務、隨機與置換 embedding 的 XGBoost。",
    "meps_event_blend_results": "MEPS 事件風險與臨床 XGBoost 做固定或驗證集融合，含置換控制。",
    "meps_event_ccs_group_hgb_seed777": "MEPS 真實住院事件連接 CCS 疾病群組率；同一預測已由論文公平比較表收錄。",
    "meps_event_ccs_soft_seed777": "以 ICD 前綴及上層群組平滑 MEPS 事件 CCS 訊號，含置換控制。",
    "meps_event_rank_correction_seed777": "只對較高風險患者套用 MEPS 事件排名校正，另比較固定與驗證集門控強度。",
    "meps_event_transfer_results": "以 MEPS 日期事件預訓練風險網路，再遷移至醫院主模型。",
    "meps_group_prior_hgb_seed777": "比較醫院內部疾病群組先驗與 MEPS 事件群組先驗。",
    "meps_harmonized_weighted_residual_seed777": "以醫院與 MEPS 分布權重調整外部風險，再校正 HGB 殘差。",
    "meps_hgb_results": "HGB 基線與加入 MEPS 風險的 HGB。",
    "meps_hgb_seed2024": "HGB 方法以另一個 seed 檢查結果穩定性。",
    "meps_hgb_seed3407": "HGB 方法以另一個 seed 檢查結果穩定性。",
    "meps_hgb_xgb_ensemble": "將三個 seed 的 out-of-fold HGB／XGBoost 機率逐筆平均。",
    "meps_longitudinal_aligned_results": "MEPS 相鄰年度對齊的預訓練／多任務編碼器。",
    "meps_longitudinal_blend_results": "2008–2012 MEPS 縱向多任務風險與醫院 XGBoost 融合，含置換控制。",
    "meps_longitudinal_blend_2008_2016_results": "把縱向融合使用的 MEPS 年份擴至 2016。",
    "meps_longitudinal_blend_2008_2024_results": "把縱向融合使用的 MEPS 年份擴至 2024。",
    "meps_longitudinal_blend_ensemble_results": "將多個縱向風險模型的 OOF 融合。",
    "meps_longitudinal_blend_seed2024": "同一 99,340 筆五折上換縱向模型隨機 seed。",
    "meps_longitudinal_blend_seed3407": "同一 99,340 筆五折上換縱向模型隨機 seed。",
    "meps_longitudinal_results": "早期 MEPS 縱向共享編碼器、預訓練及多任務模型的完整五折。",
    "meps_longitudinal_risk_xgb_results": "將 MEPS 縱向預訓練風險直接作 XGBoost 輸入。",
    "meps_longitudinal_tuned_blend_results": "以驗證資料調整縱向風險與臨床預測的融合權重。",
    "meps_longitudinal_xgb_results": "比較 MEPS 對齊預訓練／多任務 embedding 與 XGBoost。",
    "meps_multisource_blend_results": "合併 MEPS 年度縱向與住院事件外部任務。",
    "meps_multitask_results": "UCI 與 MEPS 多任務學習、預訓練及醫院 MLP 對照。",
    "meps_nested_hgb_blend_seed777": "內層驗證挑 HGB、MEPS XGBoost 與融合權重。",
    "meps_nhanes_dual_external_seed777": "MEPS 與 NHANES 單獨／合併的分組統計和多任務接法。",
    "meps_residual_correction_seed777": "用 MEPS 外部風險對醫院 HGB 預測做殘差校正。",
    "meps_utilization_results": "比較 MEPS 2、6、8 年期 ER／住院利用率訊號。",
    "real_external_sources_hgb_seed777": "擴大 MEPS 年份，並加入 NHANES 糖尿病住院訊號。",
}


def top_decile(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    alerted = np.zeros(len(y), dtype=bool)
    alerted[np.argsort(-p, kind="mergesort")[:max(1, int(np.floor(.10 * len(y))))]] = True
    tp = int(np.sum(alerted & (y == 1)))
    precision = tp / int(alerted.sum())
    recall = tp / int(y.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"AP": average_precision_score(y, p), "AUROC": roc_auc_score(y, p),
            "Precision": precision, "Recall": recall, "F1": f1,
            "Brier": brier_score_loss(y, p)}


def score(frame, probability):
    per_fold = [top_decile(part.y_true, part[probability])
                for _, part in frame.groupby("Fold", sort=True)]
    result = {key: float(np.mean([item[key] for item in per_fold])) for key in METRIC_COLS}
    result["AP_SD"] = float(np.std([item["AP"] for item in per_fold], ddof=1))
    result["Pooled_AP"] = float(average_precision_score(frame.y_true, frame[probability]))
    return result


def source_path(folder):
    return next((p for p in (folder / "oof_all.csv.gz", folder / "oof_all.csv") if p.exists()), None)


def read_oof(folder, reference):
    if folder.name == "classic_5fold_recheck_seed777":
        frame = pd.read_csv(folder / "oof_long.csv")
        keys = ["encounter_id", "patient_nbr", "y_true", "Fold"]
        frame = frame.pivot(index=keys, columns="Model", values="probability").reset_index()
        frame.columns.name = None
    else:
        path = source_path(folder)
        if path is None:
            return None
        frame = pd.read_csv(path)
    if "Fold" not in frame:
        if folder.name != "meps_hgb_xgb_ensemble":
            return None
        frame = frame.merge(reference[["encounter_id", "Fold"]], on="encounter_id", validate="one_to_one")
    keys = ["encounter_id", "patient_nbr", "y_true", "Fold"]
    if any(key not in frame for key in keys):
        return None
    if frame.encounter_id.duplicated().any() or set(frame.Fold) != {1, 2, 3, 4, 5}:
        return None
    if frame.groupby("patient_nbr").Fold.nunique().max() != 1:
        return None
    if not frame.y_true.isin([0, 1]).all():
        return None
    return frame


def group_name(folder, frame, reference, old_reference):
    columns = ["encounter_id", "patient_nbr", "y_true", "Fold"]
    sorted_keys = frame[columns].sort_values("encounter_id").reset_index(drop=True)
    if folder.name == "meps_hgb_xgb_ensemble":
        return "跨 seed OOF 平均"
    if len(frame) == len(reference):
        if sorted_keys.equals(reference):
            return "近期 99,343 筆・seed 777"
        return "99,343 筆・其他 seed"
    if len(frame) == len(old_reference):
        if sorted_keys.equals(old_reference):
            return "99,340 筆・相同折別"
        return "99,340 筆・其他 seed"
    return None


def prediction_columns(folder, frame):
    keys = {"encounter_id", "patient_nbr", "y_true", "Fold"}
    if folder.name == "meps_30d_joint_transfer_seed777":
        return [col for col in frame if col not in keys | {"meps_teacher_risk", "meps_ccs_log_support"}]
    if folder.name in {"paper_fair_comparison_seed777", "classic_5fold_recheck_seed777"}:
        return [col for col in frame if col not in keys]
    if folder.name == "cdc_clinical_interpretability_results":
        return [col for col in frame if col.startswith("prob_")]
    if folder.name == "meps_hgb_xgb_ensemble":
        return [f"p_{name}" for name in ("hgb", "clinical_xgb", "meps_xgb", "no_external", "external")]
    return [col for col in frame if col.startswith("p_")]


def inventory_oof():
    reference = pd.read_csv(RESULTS / "meps_ccs_group_hgb_seed777/oof_all.csv",
                            usecols=["encounter_id", "patient_nbr", "y_true", "Fold"])
    old_reference = pd.read_csv(RESULTS / "meps_longitudinal_blend_results/oof_all.csv",
                                usecols=["encounter_id", "patient_nbr", "y_true", "Fold"])
    reference = reference.sort_values("encounter_id").reset_index(drop=True)
    old_reference = old_reference.sort_values("encounter_id").reset_index(drop=True)
    dirs = {p.name: p for p in RESULTS.iterdir() if p.is_dir()}
    order = PREFERRED + sorted(set(dirs) - set(PREFERRED))
    seen = {}
    rows, skipped, duplicates = [], [], []
    for name in order:
        if name not in dirs or name in SKIP or name.startswith("tabpfn"):
            continue
        folder = dirs[name]
        if source_path(folder) is None and name != "classic_5fold_recheck_seed777":
            continue
        frame = read_oof(folder, reference)
        if frame is None:
            skipped.append((name, "OOF 缺少完整可核對的病人互斥五折"))
            continue
        cohort = group_name(folder, frame, reference, old_reference)
        if cohort is None:
            skipped.append((name, "隊列不屬於兩套完整五折"))
            continue
        sorted_rows = np.argsort(frame.encounter_id.to_numpy(), kind="mergesort")
        for col in prediction_columns(folder, frame):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                skipped.append((f"{name}/{col}", "預測機率缺失或超出 [0,1]"))
                continue
            fingerprint = hashlib.sha256(np.round(values[sorted_rows], 12).tobytes()).hexdigest()
            key = (cohort, fingerprint)
            if key in seen:
                duplicates.append((name, col, seen[key]))
                continue
            seen[key] = f"{name}/{col}"
            computed = score(frame, col)
            rows.append({"Cohort": cohort, "Experiment": name,
                         "Model": col.removeprefix("prob_").removeprefix("p_"),
                         **computed, "Source": f"results/{name}/{source_path(folder).name if source_path(folder) else 'oof_long.csv'}"})
        print(f"{name}: {cohort}, {len(frame):,} rows", flush=True)
    return pd.DataFrame(rows), skipped, duplicates


def summary_only_cdc():
    rows = []
    seen = set()
    for folder in sorted(RESULTS.glob("cdc*")):
        if not folder.is_dir() or (folder / "oof_all.csv").exists() or "smoke" in folder.name:
            continue
        source = next((p for p in (folder / "summary.csv", folder / "summary_mean_std.csv") if p.exists()), None)
        folds = folder / "fold_results.csv"
        if source is None or not folds.exists():
            continue
        table, fold_table = pd.read_csv(source), pd.read_csv(folds)
        if "Fold" not in fold_table or set(fold_table.Fold) != {1, 2, 3, 4, 5}:
            continue
        ap_name = "AP_mean" if "AP_mean" in table else "PR_AUC_AP_mean"
        fields = [ap_name, "AUROC_mean", "Precision_mean", "Recall_mean", "F1_mean", "Brier_mean"]
        if any(field not in table for field in fields):
            continue
        for _, record in table.iterrows():
            name = str(record.get("Experiment", record.get("FeatureSet", "model")))
            if "IH" in record and pd.notna(record.IH):
                name += f"/{record.IH}"
            numbers = [float(record[field]) for field in fields]
            if not np.isfinite(numbers).all():
                continue
            fingerprint = tuple(round(v, 10) for v in numbers)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append({"Experiment": folder.name, "Model": name,
                         **dict(zip(METRIC_COLS, numbers)), "Source": f"results/{folder.name}/{source.name}"})
    return pd.DataFrame(rows)


def table(frame, fields):
    if frame.empty:
        return "（無可核對結果。）\n"
    header = "| " + " | ".join(fields) + " |\n"
    rule = "|" + "|".join("---:" if field in METRIC_COLS + ["AP_SD", "Pooled_AP"] else "---"
                          for field in fields) + "|\n"
    lines = []
    for _, row in frame.iterrows():
        cells = []
        for field in fields:
            value = row[field]
            if field in METRIC_COLS + ["AP_SD", "Pooled_AP"]:
                cells.append(f"{value:.6f}")
            elif field == "Experiment":
                cells.append(f"[{value}](results/{value}/summary.csv)")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |\n")
    return header + rule + "".join(lines)


def shap_section():
    main = pd.read_csv(RESULTS / "paper_fair_comparison_seed777/shap_global.csv")
    rows = []
    for name in ("catboost_family", "catboost_raw", "ccs_meps_refit_hgb"):
        subset = main[main.Model.eq(name)].sort_values("Rank").head(5)
        for _, item in subset.iterrows():
            rows.append({"Model": name, "Rank": int(item.Rank), "Feature": item.Feature,
                         "MeanAbsSHAP": float(item.MeanAbsSHAP_logodds)})
    shap_table = table(pd.DataFrame(rows).rename(columns={"MeanAbsSHAP": "AP"}),
                       ["Model", "Rank", "Feature", "AP"]).replace("| AP |", "| 平均絕對 SHAP（log-odds） |")
    meps = main[main.Model.eq("ccs_meps_refit_hgb")]
    share = meps[meps.Feature.str.startswith("meps_")].MeanAbsSHAP_logodds.sum() / meps.MeanAbsSHAP_logodds.sum()
    return f"""## SHAP：模型主要依據什麼預測？

五個外層測試折各抽 1,000 筆，CatBoost 使用 TreeSHAP；HGB 的 one-hot 歸因加總回原始欄位。數值單位為模型 **raw log-odds**，排序依平均絕對 SHAP；它表示預測受該欄位影響的平均幅度，不表示風險方向或因果效果。[完整全域值](results/paper_fair_comparison_seed777/shap_global.csv)、[各折值](results/paper_fair_comparison_seed777/shap_fold.csv)。

{shap_table}
三個模型的第一、二名都是過去住院次數 `number_inpatient` 與出院去向 `discharge_disposition_id`。原始 ICD CatBoost 的 `diag_1`、`diag_3` 排第三、四，顯示診斷表示有影響；MEPS HGB 的 `meps_ccs_revisit_mean` 排第五、平均絕對 SHAP 0.040843。所有 `meps_` 特徵合計約占該模型平均絕對 SHAP 的 **{share:.1%}**，說明模型確實用到外部訊號，但配對 AP 沒有證明其帶來最終改善。`MeanSHAP_logodds` 的正負平均可能互相抵銷，不能用全域平均判斷單一特徵增加時風險的方向。

[MEPS 30 天共同訓練的 CatBoost SHAP](results/meps_30d_joint_transfer_seed777/shap_global.csv)另對每折 500 筆、共 2,500 筆測試就醫解釋：過往住院次數 0.230564、出院去向 0.146597 仍居前二；`meps_30d_teacher_risk` 為 0.032326、排第 8，`meps_ccs_log_support` 為 0.019019、排第 15。這同樣證明訊號被使用，**不等於**加入外部資料提高 AP。CDC 的[診斷感知 SHAP](results/cdc_clinical_interpretability_results/shap_importance.csv)亦以住院次數及出院去向為主；[DiffWalk 控制](results/cdc_diffwalk_control_results/cdc_shap_summary.csv)有外部特徵歸因，但不單靠歸因宣稱提升。

SHAP 只針對上述已保存解釋檔的模型分析；其他表格中的模型沒有可核對的 SHAP 結果，不推測其特徵排序。

"""


def three_external_section():
    """Summarize the completed CMS + MEPS + CDC experiment from saved results."""
    folder = RESULTS / "cms_meps_cdc_catboost_seed777"
    summary = pd.read_csv(folder / "summary.csv")
    bootstrap = pd.read_csv(folder / "paired_ap_bootstrap.csv")
    shap = pd.read_csv(folder / "shap_global.csv")
    thresholds = pd.read_csv(RESULTS / "top_fractions_cms_meps_cdc_5fold_seed777" / "summary.csv")
    required = {"catboost_raw", "cms_catboost_raw", "ccs_only_hgb_catboost_raw_50", "meps", "cdc",
                "cms_meps", "cms_cdc", "meps_cdc", "cms_meps_cdc", "cms_meps_cdc_hgb50"}
    if not required.issubset(set(summary.Model)) or not summary.Folds.eq(5).all():
        raise ValueError("Three-source experiment lacks complete five-fold metrics")
    names = ["catboost_raw", "ccs_only_hgb_catboost_raw_50", "cms_catboost_raw", "meps", "cdc",
             "cms_meps", "cms_cdc", "meps_cdc", "cms_meps_cdc", "cms_meps_cdc_hgb50"]
    metrics = summary.set_index("Model").loc[names]
    rows = ["| 模型 | PR-AUC | AUROC | Precision | Recall | F1 | Brier |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    columns = ["PR_AUC_mean", "AUROC_mean", "Precision_mean", "Recall_mean", "F1_mean", "Brier_mean"]
    for name, record in metrics.iterrows():
        rows.append("| `" + name + "` | " + " | ".join(f"{record[col]:.6f}" for col in columns) + " |")
    comparison = bootstrap.loc[
        bootstrap.Treatment.eq("cms_meps_cdc_hgb50") &
        bootstrap.Control.eq("ccs_only_hgb_catboost_raw_50")
    ]
    if len(comparison) != 1:
        raise ValueError("Missing paired bootstrap for three-source ensemble")
    paired = comparison.iloc[0]
    external_names = ["meps_ccs_revisit_mean", "meps_ccs_readmit_max", "meps_ccs_revisit_max",
                      "meps_ccs_readmit_mean", "cdc_poor_health", "cms_readmission_rate", "cms_mapped"]
    selected = shap.set_index("Feature").loc[external_names]
    if not selected.Explained_Rows.eq(5000).all():
        raise ValueError("Three-source SHAP explanation sample changed")
    shap_rows = ["| 外部特徵 | 平均絕對 SHAP（log-odds） | 排名 |",
                 "| --- | ---: | ---: |"]
    for feature, record in selected.iterrows():
        shap_rows.append(f"| `{feature}` | {record.MeanAbsSHAP_logodds:.6f} | {int(record.Rank)} |")
    target = thresholds.loc[thresholds.Model.isin(["ccs_only_catboost_raw_50", "cms_meps_cdc_hgb50"])]
    f1 = target.pivot(index="Top_Percent", columns="Model", values="F1")
    if set(f1.index) != {5., 10., 15., 20., 25.}:
        raise ValueError("Three-source alert-budget results incomplete")
    budget_rows = ["| 每折警示比例 | 無外部資料融合 F1 | 三來源融合 F1 |",
                   "| --- | ---: | ---: |"]
    for fraction, values in f1.sort_index().iterrows():
        budget_rows.append(f"| Top {fraction:g}% | {values['ccs_only_catboost_raw_50']:.6f} | "
                           f"{values['cms_meps_cdc_hgb50']:.6f} |")
    return f"""## CDC＋MEPS＋CMS 三來源聯合實驗

這組已完成 **99,343 筆、69,990 位病人、seed 777、相同病人互斥五折**。所有模型的 Precision、Recall、F1 都固定取每折風險最高 10%；其餘三項使用原始預測機率。單一及雙來源列使用同一 Raw ICD-9 CatBoost 架構做配對控制；`cms_meps_cdc_hgb50` 再與 CCS-only HGB 的機率固定 50/50 融合。下表中的 `ccs_only_hgb_catboost_raw_50` 與前文 `ccs_only_catboost_raw_50` 是同一基線預測。[完整六項指標](results/cms_meps_cdc_catboost_seed777/summary.csv)｜[逐折指標](results/cms_meps_cdc_catboost_seed777/fold_metrics.csv)｜[實驗稽核](results/cms_meps_cdc_catboost_seed777/audit.json)。

{chr(10).join(rows)}

三來源融合相對無外部資料的 CCS HGB＋Raw CatBoost，平均 **ΔAP {paired.MeanFold_Delta_PR_AUC:+.6f}**；每折內按病人配對抽樣 2,000 次的 95% CI 為 **[{paired.CI95_Lower:+.6f}, {paired.CI95_Upper:+.6f}]**。Top 10% 的 F1 從 **{metrics.loc['ccs_only_hgb_catboost_raw_50','F1_mean']:.6f}** 到 **{metrics.loc['cms_meps_cdc_hgb50','F1_mean']:.6f}**，但 AP 較低且配對 CI 跨零，不能宣稱三來源聯合提高主要 PR-AUC。其他配對差值見[完整 bootstrap](results/cms_meps_cdc_catboost_seed777/paired_ap_bootstrap.csv)。

CDC 使用 BRFSS 糖尿病受訪者的**自評健康不佳代理分數**，它本身不是再住院標籤；MEPS 使用精確日期事件的 CCS 群組 30 天再就醫／再住院統計；CMS 使用四類疾病的全國彙總再住院率，約覆蓋醫院隊列 16.5%。這些欄位以共同人口或診斷資訊對齊；不同資料來源的測量方式與涵蓋範圍，可能限制外部訊號的效益，但本實驗不能判定因果原因。

聯合 CatBoost 的五折 SHAP 每折抽 1,000 筆，共 5,000 筆。原始醫院特徵 `number_inpatient`、`discharge_disposition_id`、`diag_1` 仍排前三；外部特徵如下。[完整 SHAP 排名](results/cms_meps_cdc_catboost_seed777/shap_global.csv)｜[各折 SHAP](results/cms_meps_cdc_catboost_seed777/shap_fold.csv)。SHAP 表示模型使用了特徵，不能單靠它證明外部資料增加預測效益。

{chr(10).join(shap_rows)}

同一組 OOF 的固定警示比例敏感度分析如下；**PR-AUC、AUROC、Brier 不隨警示比例改變**。三來源融合在 Top 5% 與 10% 的 F1 略高，Top 15%～25% 則沒有此優勢。[全部模型與比例的六項指標](results/top_fractions_cms_meps_cdc_5fold_seed777/summary.csv)。

{chr(10).join(budget_rows)}

"""


def main():
    inventory, skipped, duplicates = inventory_oof()
    cdc = summary_only_cdc()
    if inventory.empty:
        raise RuntimeError("No complete five-fold OOF results found")
    inventory.to_csv(RESULTS / "fivefold_metrics_recomputed.csv", index=False)
    columns = ["Experiment", "Model", "AP", "AUROC", "Precision", "Recall", "F1", "Brier", "Pooled_AP"]
    score_names = ["AP", "AUROC", "Precision", "Recall", "F1", "Brier"]
    main_set = inventory[inventory.Cohort.eq("近期 99,343 筆・seed 777")]
    paper = main_set[main_set.Experiment.eq("paper_fair_comparison_seed777")]
    best = paper.loc[paper.AP.idxmax()]
    classic = main_set[main_set.Experiment.eq("classic_5fold_recheck_seed777")]
    latest_external = main_set[main_set.Experiment.eq("meps_nhanes_dual_external_seed777")]
    best_external = paper[paper.Model.eq("ccs_meps_catboost_50")].iloc[0]
    best_dual = latest_external[latest_external.Model.str.contains("meps|nhanes|dual")].sort_values("AP", ascending=False).iloc[0]
    intro = f"""# 糖尿病 30 天再住院：已完成五折實驗總分析

更新日期：2026-11-23。本報告只使用**已完成五折**且具備六項指標的結果。主要任務是以 UCI Diabetes 130-US Hospitals 的出院時資料預測 30 天內再住院。AP／PR-AUC 是主指標；另完整列出 AUROC、Precision、Recall、F1 與 Brier。

## 方法與一致的計算口徑

- **近期主切分**：99,343 筆就醫、69,990 位病人，seed 777，病人互斥五折；此組所有 OOF 已核對 encounter、病人、標籤、折別與[參考五折](results/meps_ccs_group_hgb_seed777/oof_all.csv)完全一致。不同 seed 和較早 99,340 筆五折分表，不能混成同一排行。
- 所有有完整 OOF 的模型都從**測試折機率**重算六項指標。每折依風險排序，固定選最高的 10% 為警示對象，再計算 Precision、Recall、F1；沒有利用測試標籤選門檻。AP 採 `average_precision_score`；AUROC 和 Brier 不依賴門檻。六項指標報五折算術平均；另列合併 OOF 的 pooled AP，兩者不是同一統計量。
- 部分 CDC 五折只有六項指標摘要、沒有可重算的完整 OOF。其表保留**原執行的驗證集門檻**，Precision／Recall／F1 不與固定最高 10% 表直接相比；AP、AUROC、Brier 可以比較，但沒有逐筆切分稽核。
- MEPS、CDC BRFSS、NHANES 是真實外部病人或調查資料；AHRQ CCS 是疾病分類表，CCS-only **不算**接入外部病人資料。不同外部來源的標籤與 UCI 30 天再住院可能不一致。

## 主要發現

1. 近期同一五折的最佳平均 AP 為 **{best.AP:.6f}**（{best.Model}）；六項指標見下表。含真實外部病人資料的最佳方法為 **{best_external.AP:.6f}**（{best_external.Model}）；MEPS＋NHANES 雙來源方法中最高為 **{best_dual.AP:.6f}**（{best_dual.Model}）。這只代表已完成模型的數值排名，並不證明外部資料產生增益。
2. 新五折 LR／RF／XGBoost 基線中，標準 XGBoost **{classic.loc[classic.Model.eq('xgb_standard'),'AP'].iloc[0]:.6f}** 最好；class weight 與 SMOTENC 都未提高 AP。與早期單次 holdout 數字不混用。
3. MEPS 30 天共同訓練對同架構 UCI-only 神經網路有小幅正差；但最高 AP 的 HGB＋CatBoost 融合中，MEPS、NHANES 群組或多任務版本都沒有穩定超過不含外部資料的對照。必須把「有接入真實 external data」與「external data 額外改善 AP」分開陳述。

## 論文復現、標準基線與主要外部資料結果

下表均屬近期相同五折，六項指標使用同一個最高 10% 警示規則；重複的 OOF 機率只列一次。論文方法的 `bhuvan_hgb` 是在**本專題五折**重跑的復現對照，不是搬用論文單次測試數字。

### 論文復現、CatBoost 與 MEPS 融合

{table(paper.sort_values('AP', ascending=False), columns)}

同一折病人配對 bootstrap 顯示，`catboost_raw` 相對 `ccs_only_hgb` 的平均 AP 差為 **+0.007327**（95% CI **[+0.004039, +0.010484]**）。但 `ccs_meps_catboost_50` 相對同架構 `ccs_only_catboost_50` 為 **−0.000197**（95% CI **[−0.001026, +0.000604]**），加入 MEPS 的額外增益未獲證實。[配對分析](results/paper_fair_comparison_seed777/paired_ap_bootstrap.csv)。

### LR、RF、XGBoost 與類別不平衡

{table(classic.sort_values('AP', ascending=False), columns)}
標準 XGBoost 對標準 RF、標準 LR 的平均 AP 差依序為 +0.008345、+0.017776；加權 XGBoost 相對標準 XGBoost 為 −0.009276。SMOTENC 的 0.20、0.35、0.50、1.00 比例越高，AP 依序約 0.206、0.196、0.186、0.172。加權 LR／RF／XGBoost 的 Brier 約 0.22，直接輸出的機率沒有良好校準。

### MEPS＋NHANES 雙來源與 30 天共同訓練

{table(main_set[main_set.Experiment.isin(['meps_nhanes_dual_external_seed777','meps_30d_joint_transfer_seed777'])].sort_values(['Experiment','AP'], ascending=[True,False]), columns)}
外部 30 天共同訓練 `joint_meps_30d` 相對 `uci_only_nn` 的平均 AP 差 **+0.002644**，病人配對 bootstrap 95% CI **[+0.000035, +0.005273]**；加上 CatBoost 後仍低於最佳 CCS-only 融合。雙來源最佳融合的增益相對最佳 CCS-only 融合未獲證實。[配對檔](results/meps_30d_joint_transfer_seed777/paired_ap_bootstrap.csv)、[雙來源結果](results/meps_nhanes_dual_external_seed777/summary.csv)。

"""
    others = main_set[~main_set.Experiment.isin([
        "paper_fair_comparison_seed777", "classic_5fold_recheck_seed777",
        "meps_nhanes_dual_external_seed777", "meps_30d_joint_transfer_seed777",
    ])]
    parts = [intro, "## 近期同切分其餘完整五折實驗\n\n"]
    parts.append("包含 CDC、MEPS 年度利用率／事件、CCS 群組、殘差校正、外部來源擴充等不同接法。每列均是該實驗的**獨立預測**；完全相同的 OOF 控制組已去重。數字使用與主表相同的六項指標計算。\n\n")
    for name, group in others.groupby("Experiment", sort=True):
        parts.append(f"### [{name}](results/{name}/summary.csv)\n\n{DESCRIPTIONS.get(name, '已完成五折的不同模型與外部特徵接法。')}\n\n{table(group.sort_values('AP', ascending=False), ['Model',*score_names,'Pooled_AP'])}\n")
    parts.append("## 較早 99,340 筆隊列的完整五折實驗\n\n這些仍是病人互斥五折，主要涵蓋 MEPS 縱向預訓練、風險融合與事件遷移；但隊列少 3 筆且折別與近期主表不同，**只在本節內描述**。全部從完整 OOF 以相同最高 10% 警示規則重算六項指標；不直接拿來和 99,343 筆表判斷勝負。\n\n")
    old = inventory[inventory.Cohort.eq("99,340 筆・相同折別")]
    for name, group in old.groupby("Experiment", sort=True):
        parts.append(f"### [{name}](results/{name}/summary.csv)\n\n{DESCRIPTIONS.get(name, '已完成五折的不同模型與外部特徵接法。')}\n\n{table(group.sort_values('AP', ascending=False), ['Model',*score_names,'Pooled_AP'])}\n")
    parts.append("## 其他 seed 與跨 seed OOF 平均\n\n以下每個 seed 各自完成病人互斥五折，卻不使用 seed 777 的相同測試折；跨 seed 平均則是逐筆多個 OOF 預測的平均。用來檢查穩定性，不併入主表排名。\n\n")
    other_seeds = inventory[~inventory.Cohort.isin(["近期 99,343 筆・seed 777","99,340 筆・相同折別"])]
    for cohort, cohort_group in other_seeds.groupby("Cohort", sort=True):
        parts.append(f"### {cohort}\n\n{table(cohort_group.sort_values(['Experiment','AP'], ascending=[True,False]), columns)}\n")
    parts.append("## CDC 摘要型五折實驗\n\n這些來源保存五折六項指標，但沒有可逐筆重算全部模型的 OOF；因此採原摘要的驗證集選門檻，Precision／Recall／F1 與上表的固定最高 10% 不同。重複的完全相同摘要行已去重。\n\n")
    for name, group in cdc.groupby("Experiment", sort=True):
        source = group.Source.iloc[0]
        parts.append(f"### [{name}]({source})\n\n{DESCRIPTIONS.get(name, 'CDC 外部健康資料的完整五折摘要。')}\n\n{table(group.sort_values('AP', ascending=False), ['Model',*score_names])}\n")
    unique_folders = set(inventory.Experiment)
    fully_repeated = sorted({name for name, _, _ in duplicates if name not in unique_folders})
    if fully_repeated:
        parts.append("## 重複結果的來源\n\n下列實驗已完成五折，但其保存的預測與上表某個模型完全一致；為避免同一模型重複算入，僅保留較前面的主表來源：\n\n")
        for name in fully_repeated:
            equivalents = sorted({canonical for folder, _, canonical in duplicates if folder == name})
            parts.append(f"- [{name}](results/{name}/summary.csv)：對應 `{', '.join(equivalents)}`。\n")
        parts.append("\n")
    parts.append(shap_section())
    parts.append("""## 統計解讀與報告界線

在同一測試折比較模型時，優先看配對病人 bootstrap；單看五折平均或五折標準差不能證明差異。主表結果經多次嘗試與選模，仍有探索性選模偏差。真實外部資料的成功接入可由特徵來源與 SHAP 證明，但若和同架構、同切分、無外部資料對照的 AP 差沒有穩定正值，就不能宣稱外部資料提升主要任務。本文只報已完成的五折；不填補未產生的數字。

本報告可由 `python scripts/experiments/build_results_analysis.py` 依目前保存的 OOF 和 CDC 摘要重新生成；重算的逐方法六指標另存於 `results/fivefold_metrics_recomputed.csv`。\n""")
    destination = ROOT / "RESULTS_ANALYSIS.md"
    historical = "".join(parts)
    final_report = RESULTS / "final_presentation_completion" / "FINAL_EXPERIMENT_SUMMARY.md"
    if final_report.exists():
        marker = "## 論文復現、標準基線與主要外部資料結果"
        if marker not in historical:
            raise ValueError("Historical report structure changed; cannot safely combine reports")
        historical = historical[historical.index(marker):]
        historical = historical.replace(marker, "## 先前論文復現、標準基線與外部資料延伸", 1)
        historical = historical.replace(
            "本報告可由 `python scripts/experiments/build_results_analysis.py`",
            "本附錄可由 `python scripts/experiments/build_results_analysis.py`",
        )
        current = final_report.read_text(encoding="utf-8")
        current = current.replace(
            "# Final Presentation 實驗統整：30 天糖尿病再住院",
            "# 糖尿病 30 天再住院：五折實驗與完整結果分析",
            1,
        )
        current = current.replace(
            "產生方式：",
            "更新日期：2026-11-24。正式比較限於同一 99,343 筆、seed 777 病人互斥五折；CDC、MEPS、CMS 為正式 external data。其他已完成實驗保留於後面的歷史附錄，不納入主排行。\n\n產生方式：",
            1,
        )
        current = current.replace(
            "本檔只依據已核對的實際 OOF 和原始資料生成；不使用早期單次 holdout、99,340 筆隊列或其他 seed。",
            "前半部正式主表只依據已核對的實際 OOF 和原始資料生成；不使用早期單次 holdout、99,340 筆隊列或其他 seed。",
            1,
        )
        current = current.replace(
            "產生方式：`python scripts/experiments/complete_final_presentation.py --bootstrap 2000`。",
            "產生方式：先執行 `python scripts/experiments/complete_final_presentation.py --bootstrap 2000`，再執行 `python scripts/experiments/build_results_analysis.py` 整合歷史附錄。",
            1,
        )
        current = current.replace(
            "詳細設定見 `catboost_ablation_run.json`，原始模型設定見 `results/paper_fair_comparison_seed777/run.json`。",
            "詳細設定見 [CatBoost 消融設定](results/final_presentation_completion/catboost_ablation_run.json)，原始模型設定見 [論文比較設定](results/paper_fair_comparison_seed777/run.json)。",
            1,
        )
        current = current.replace(
            "## 研究協定與可重用資料",
            "主要輸出：[六項指標](results/final_presentation_completion/final_experiment_metrics.csv)、[消融指標](results/final_presentation_completion/final_ablation_metrics.csv)、[配對 bootstrap](results/final_presentation_completion/final_bootstrap_comparisons.csv)、[CMS coverage](results/final_presentation_completion/cms_coverage_analysis.csv)、[CDC＋MEPS＋CMS 聯合結果](results/cms_meps_cdc_catboost_seed777/summary.csv)、[OOF 稽核](results/final_presentation_completion/source_audit.csv)。\n\n## 研究協定與可重用資料",
            1,
        )
        external_path = RESULTS / "final_presentation_completion" / "external_data_comparisons.csv"
        external = pd.read_csv(external_path)
        paired_rows = ["| 來源 | 原始模型 | 外部模型 | ΔAP | 95% CI |",
                       "| --- | --- | --- | ---: | ---: |"]
        for row in external.itertuples(index=False):
            paired_rows.append(
                f"| {row.External_Source} | `{row.Control}` | `{row.Treatment}` | "
                f"{row.Delta_PR_AUC:+.6f} | [{row.CI95_Lower:+.6f}, {row.CI95_Upper:+.6f}] |"
            )
        paired_table = "\n".join(paired_rows)
        current = current.replace(
            "## External Data：觀察、可能原因與未驗證假設\n\n",
            "## External Data：觀察、可能原因與未驗證假設\n\n"
            "下表只做同架構配對比較；兩個模型各自的六項指標見前述正式表，"
            "[完整配對表](results/final_presentation_completion/external_data_comparisons.csv)"
            "另含兩者的六項數值與對齊方式。\n\n" + paired_table + "\n\n### 資料來源、對齊方式與限制\n\n",
            1,
        )
        marker = "## CatBoost 原始特徵消融"
        if marker not in current:
            raise ValueError("Final presentation report changed; cannot insert three-source results")
        current = current.replace(marker, three_external_section() + marker, 1)
        current += "\n# 延伸實驗與歷史紀錄\n\n以下保留先前已完成的其他五折方法，包括 NHANES、不同 seed 與 99,340 筆舊隊列。這些結果供方法歷程查閱；不同隊列、切分或門檻策略不得與前述正式主表直接排名。重複的核心模型以最前方的最新 OOF 重算表為準。\n\n"
        destination.write_text(current + historical, encoding="utf-8")
    else:
        destination.write_text(historical, encoding="utf-8")
    print(f"Saved {destination}: {len(inventory)} OOF model rows, {len(cdc)} CDC summary rows")
    if skipped:
        print("Skipped:", skipped)


if __name__ == "__main__":
    main()
