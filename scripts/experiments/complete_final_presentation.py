#!/usr/bin/env python3
"""Audit existing OOFs and complete reproducible final-presentation tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.paper_comparison import top_fraction_metrics  # noqa: E402
from run_cms_group_prior_5fold import checksum, read_cms  # noqa: E402
from run_dual_external_meps_nhanes import hospital_frame  # noqa: E402
from run_paper_fair_comparison import paired_patient_bootstrap  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
PATHS = {
    "paper": "results/paper_fair_comparison_seed777/oof_all.csv.gz",
    "cms": "results/cms_group_prior_5fold_seed777/oof_all.csv.gz",
    "cdc": "results/cdc_clinical_interpretability_results/oof_all.csv",
    "diffwalk": "results/cdc_diffwalk_control_results/oof_all.csv",
    "meps_nn": "results/meps_30d_joint_transfer_seed777/oof_all.csv.gz",
}
MODEL_SPECS = [
    ("bhuvan_hgb", "paper", "bhuvan_hgb", "Baseline"),
    ("roy_hgb_46", "paper", "roy_hgb_46", "Baseline"),
    ("ccs_only_hgb", "paper", "ccs_only_hgb", "Baseline"),
    ("catboost_raw", "paper", "catboost_raw", "Baseline"),
    ("ccs_only_catboost_raw_50", "paper", "ccs_only_catboost_raw_50", "Baseline"),
    ("ccs_meps_group_hgb", "paper", "ccs_meps_group_hgb", "MEPS"),
    ("ccs_meps_catboost_raw_50", "paper", "ccs_meps_catboost_raw_50", "MEPS"),
    ("uci_only_nn", "meps_nn", "uci_only_nn", "MEPS"),
    ("joint_meps_30d", "meps_nn", "joint_meps_30d", "MEPS"),
    ("joint_meps_30d_catboost50", "meps_nn", "joint_meps_30d_catboost50", "MEPS"),
    ("E2_Clinical", "cdc", "prob_E2_Clinical", "CDC"),
    ("E4_Clinical_CDCHealth", "cdc", "prob_E4_Clinical_CDCHealth", "CDC"),
    ("Only_DiffWalk", "cdc", "prob_Only_DiffWalk", "CDC"),
    ("No_DiffWalk", "cdc", "prob_No_DiffWalk", "CDC"),
    ("Permuted_DiffWalk", "diffwalk", "p_Permuted_DiffWalk", "CDC"),
    ("cms_hgb", "cms", "cms_hgb", "CMS"),
    ("cms_catboost_raw", "cms", "cms_catboost_raw", "CMS"),
    ("cms_hgb_catboost_raw_50", "cms", "cms_hgb_catboost_raw_50", "CMS"),
    ("ccs_only_hgb_cms_catboost_50", "cms", "ccs_only_hgb_cms_catboost_50", "CMS"),
    ("cms_hgb_cms_catboost_50", "cms", "cms_hgb_cms_catboost_50", "CMS"),
]
BASELINE_AP = {"bhuvan_hgb": .22870317444605384, "roy_hgb_46": .23726129364736193,
               "ccs_only_hgb": .2391789277933903, "catboost_raw": .2465063597458621,
               "ccs_only_catboost_raw_50": .247373509021748}
METRIC_COLUMNS = ["PR-AUC", "AUROC", "Precision", "Recall", "F1", "Brier"]


def load_sources() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[dict]]:
    sources = {}
    audit = []
    for name, relative in PATHS.items():
        path = ROOT / relative
        data = pd.read_csv(path)
        if any(key not in data for key in KEYS):
            raise ValueError(f"Missing OOF keys: {path}")
        if len(data) != 99343 or not data.encounter_id.is_unique or set(data.Fold) != {1, 2, 3, 4, 5}:
            raise ValueError(f"Not the 99,343-row five-fold cohort: {path}")
        if data.patient_nbr.nunique() != 69990 or data.groupby("patient_nbr").Fold.nunique().max() != 1:
            raise ValueError(f"Patient-disjoint audit failed: {path}")
        sources[name] = data
        audit.append({"Source": name, "Path": str(path), "Rows": len(data),
                      "Patients": int(data.patient_nbr.nunique()), "SHA256": checksum(path)})
    reference = sources["paper"][KEYS].copy()
    for name, data in sources.items():
        aligned = data.set_index("encounter_id").loc[reference.encounter_id].reset_index()
        if not aligned[KEYS].equals(reference):
            raise ValueError(f"Encounter, patient, label or fold mismatch: {name}")
        sources[name] = aligned
    return sources, reference, audit


def summarize(scores: pd.DataFrame, model_sources: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for fold, part in scores.groupby("Fold", sort=True):
        for name in model_sources:
            result = top_fraction_metrics(part.y_true, part[name], .10)
            fold_rows.append({"Fold": int(fold), "Model": name, "Source": model_sources[name],
                              "N": len(part), **result})
    folds = pd.DataFrame(fold_rows)
    records = []
    for name, group in folds.groupby("Model", sort=False):
        records.append({
            "Model": name, "Source": model_sources[name], "Folds": len(group),
            "PR-AUC": float(group.PR_AUC.mean()), "AUROC": float(group.AUROC.mean()),
            "Precision": float(group.Precision.mean()), "Recall": float(group.Recall.mean()),
            "F1": float(group.F1_score.mean()), "Brier": float(group.Brier_Score.mean()),
            "Pooled_OOF_AP": float(average_precision_score(scores.y_true, scores[name])),
            "Threshold": "top 10% per test fold", "Cohort_Rows": len(scores),
        })
    return pd.DataFrame(records), folds


def delta(scores: pd.DataFrame, treatment: str, control: str) -> float:
    return float(np.mean([
        average_precision_score(part.y_true, part[treatment])
        - average_precision_score(part.y_true, part[control])
        for _, part in scores.groupby("Fold", sort=True)
    ]))


def existing_bootstrap(scores: pd.DataFrame, file: Path, pairs: list[tuple[str, str]], label: str) -> pd.DataFrame:
    table = pd.read_csv(file)
    table["Original_Control"] = table.Control
    table["Control"] = table.Control.replace({
        "ccs_only_hgb_catboost_raw_50": "ccs_only_catboost_raw_50"})
    subset = table.loc[table.apply(lambda row: (row.Treatment, row.Control) in pairs, axis=1)].copy()
    if len(subset) != len(pairs):
        raise ValueError(f"Missing expected saved bootstrap comparison in {file}")
    for _, row in subset.iterrows():
        if not np.isclose(row.MeanFold_Delta_PR_AUC, delta(scores, row.Treatment, row.Control), atol=1e-12, rtol=0):
            raise ValueError(f"Saved bootstrap observed delta disagrees: {file}, {row.Treatment}")
    subset["Source"] = label
    subset["Reused"] = True
    return subset


def cms_coverage(args, scores: pd.DataFrame) -> pd.DataFrame:
    # Rebuild the *same* principal-CCS mapping used by the original CMS experiment.
    paper = scores[KEYS].copy()
    hospital, _ = hospital_frame(SimpleNamespace(hospital=args.hospital, ccs_file=args.ccs_file), paper)
    _, rates = read_cms(args.cms_prior)
    matched = hospital.ccs_diag_1.isin(rates).to_numpy()
    declared = json.loads((args.cms_prior.parent / "cms_readmission_ccs_prior.audit.json").read_text())
    if int(matched.sum()) != declared["uci_principal_covered_rows"]:
        raise ValueError("CMS coverage disagrees with the original preparation audit")
    records = []
    for label, mask in (("CMS-covered", matched), ("CMS-uncovered", ~matched)):
        group = scores.loc[mask]
        for model in ("ccs_only_hgb", "cms_hgb", "catboost_raw", "cms_catboost_raw",
                      "ccs_only_catboost_raw_50", "ccs_only_hgb_cms_catboost_50"):
            part_metrics = [top_fraction_metrics(part.y_true, part[model], .10)
                            for _, part in group.groupby("Fold", sort=True)]
            records.append({
                "Subset": label, "Model": model, "N": len(group),
                "Patients": int(group.patient_nbr.nunique()), "Positive_Count": int(group.y_true.sum()),
                "Positive_Rate": float(group.y_true.mean()),
                "PR-AUC": float(np.mean([m["PR_AUC"] for m in part_metrics])),
                "AUROC": float(np.mean([m["AUROC"] for m in part_metrics])),
                "Precision": float(np.mean([m["Precision"] for m in part_metrics])),
                "Recall": float(np.mean([m["Recall"] for m in part_metrics])),
                "F1": float(np.mean([m["F1_score"] for m in part_metrics])),
                "Brier": float(np.mean([m["Brier_Score"] for m in part_metrics])),
                "Pooled_OOF_AP": float(average_precision_score(group.y_true, group[model])),
                "Threshold": "top 10% within subgroup in each fold; exploratory",
            })
    return pd.DataFrame(records)


def markdown_table(table: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    header = "| " + " | ".join(
        "PR-AUC (Average Precision, AP)" if column == "PR-AUC" else column
        for column in columns
    ) + " |"
    rule = "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |"
    rows = []
    for _, row in table[columns].iterrows():
        values = [f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
                  for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *rows])


def write_report(args, metrics, ablation, bootstrap, coverage, missing, audit) -> None:
    by_name = metrics.set_index("Model")
    best = by_name.loc["ccs_only_catboost_raw_50", "PR-AUC"]
    cms_best = by_name.loc["ccs_only_hgb_cms_catboost_50", "PR-AUC"]
    raw_ensemble = by_name.loc["raw_hgb_catboost_raw_50", "PR-AUC"]
    cdc_delta = by_name.loc["E4_Clinical_CDCHealth", "PR-AUC"] - by_name.loc["E2_Clinical", "PR-AUC"]
    meps_delta = by_name.loc["joint_meps_30d", "PR-AUC"] - by_name.loc["uci_only_nn", "PR-AUC"]
    cols = ["Model", *METRIC_COLUMNS]
    groups = {
        "主要 Baseline 與 CCS 融合": ["bhuvan_hgb", "roy_hgb_46", "ccs_only_hgb", "catboost_raw",
                                   "raw_hgb_catboost_raw_50", "ccs_only_catboost_raw_50"],
        "CMS HRRP": ["cms_hgb", "cms_catboost_raw", "cms_hgb_catboost_raw_50",
                     "ccs_only_hgb_cms_catboost_50", "cms_hgb_cms_catboost_50"],
        "CDC BRFSS": ["E2_Clinical", "E4_Clinical_CDCHealth", "Only_DiffWalk",
                       "No_DiffWalk", "Permuted_DiffWalk"],
        "MEPS": ["ccs_meps_group_hgb", "ccs_meps_catboost_raw_50", "uci_only_nn",
                 "joint_meps_30d", "joint_meps_30d_catboost50"],
    }
    sections = []
    for title, names in groups.items():
        sections.append(f"## {title}\n\n" + markdown_table(metrics.set_index("Model").loc[names].reset_index(), cols))
    ablation_text = markdown_table(ablation, cols + ["Full_minus_Ablated_PR_AUC"]) if not ablation.empty else "尚未完成必做的 E1 五折訓練。"
    key_pairs = bootstrap.loc[bootstrap.Treatment.isin([
        "ccs_only_catboost_raw_50", "ccs_only_hgb_cms_catboost_50", "cms_hgb_catboost_raw_50",
        "E4_Clinical_CDCHealth", "joint_meps_30d", "catboost_raw",
    ])]
    ci = markdown_table(key_pairs, ["Treatment", "Control", "MeanFold_Delta_PR_AUC", "CI95_Lower", "CI95_Upper"])
    subgroup = coverage.loc[coverage.Model.isin(["ccs_only_catboost_raw_50", "ccs_only_hgb_cms_catboost_50"])]
    subgroup_table = markdown_table(subgroup, ["Subset", "Model", "N", "Positive_Count", "Positive_Rate", "PR-AUC"])
    shap_raw = pd.read_csv(ROOT / "results/paper_fair_comparison_seed777/shap_global.csv")
    shap_cms = pd.read_csv(ROOT / "results/cms_group_prior_5fold_seed777/shap_global.csv")
    raw_top = shap_raw.loc[shap_raw.Model.eq("catboost_raw")].sort_values("Rank").head(5)
    cms_shap = shap_cms.loc[shap_cms.Feature.isin(["cms_readmission_rate", "cms_mapped"])]
    raw_shap_table = markdown_table(raw_top, ["Feature", "MeanAbsSHAP_logodds", "Rank"])
    cms_shap_table = markdown_table(cms_shap, ["Feature", "MeanAbsSHAP_logodds", "Rank"])
    missing_text = "\n".join(f"- **{r.Item}**：{r.Status}；{r.Reason}" for r in missing.itertuples()) or "無。"
    full = f"""# Final Presentation 實驗統整：30 天糖尿病再住院

產生方式：`python scripts/experiments/complete_final_presentation.py --bootstrap {args.bootstrap}`。本檔只依據已核對的實際 OOF 和原始資料生成；不使用早期單次 holdout、99,340 筆隊列或其他 seed。

## 研究協定與可重用資料

UCI 共 **99,343** 次就醫、**69,990** 位病人，seed **777**，病人互斥五折。所有納入的 OOF 均逐筆核對 Encounter ID、Patient ID、Fold、Label。AP 為 `average_precision_score`；表列六項指標為各測試折的**算術平均**。Precision、Recall、F1 固定取每折預測風險最高 10%。合併所有 OOF 計算的 pooled AP 存在 CSV 的另一欄，不與五折平均混淆。對照資料及 SHA256 見 `source_audit.csv`。

CCS 是 ICD-9 疾病分類表，單獨使用 CCS 不算加入患者級外部資料；正式的外部資料來源限定為 CDC BRFSS、MEPS、CMS HRRP。六項指標的方向為 PR-AUC (Average Precision, AP)、AUROC、Precision、Recall、F1 越高越好，Brier 越低越好。

先前論文報告的 0.237 AP 使用不同的單次 holdout；它只能作背景參考。公平比較以此處同一 99,343 筆、同五折 OOF 的 Rajpoot HGB 復現（AP 0.228703）及其他同切分模型為準。

已重用：論文復現 HGB、Raw HGB、CCS HGB、Raw CatBoost、MEPS、CDC、CMS 的既有五折 OOF；CMS 既有 2,000 次病人分折 bootstrap；MEPS 既有配對 bootstrap。新計算：Raw HGB＋Raw CatBoost 固定 50/50 融合、CMS covered 子群與缺少的配對 bootstrap。Ablation 只有重新訓練後才納入。

{chr(10).join(sections).replace(chr(10) + '## ', chr(10) + chr(10) + '## ')}

## CatBoost 原始特徵消融

{ablation_text}

E0 是原始 46 欄 Raw ICD-9 CatBoost，直接重用 OOF。E1 移除 `number_inpatient`；E2 只移除 `diag_1`、`diag_2`、`diag_3`；E3 另外移除由診斷資料計算的 `number_diagnoses`。此 Raw CatBoost 未含 CCS 或其他診斷衍生欄位。每組已完成的消融均沿用原本的病人互斥內層選參程序並重新訓練；未完成者只列於缺項表。

## 配對比較與不確定性

以下差值均為五折 AP 差的平均。Bootstrap 單位為每折內的病人；95% CI 不能直接解讀為來源造成改變的因果證據。完整比較見 `final_bootstrap_comparisons.csv`。

{ci}

## CMS coverage 與特徵內容

CMS 只依 UCI 主要診斷 CCS 對應四個概念：心肌梗塞、心衰竭、肺炎、COPD。命中率為 **{audit['cms_mapped_rows']:,}/99,343（{audit['cms_mapped_fraction']:.1%}）**。數值是 CMS 全國風險標準化再住院率，並非個人標籤。`ccs_only_hgb_cms_catboost_50` 是原始 **46 欄＋CCS 的 HGB**，與 **原始 46 欄＋CMS rate／mapped indicator 的 CatBoost** 以預測機率 50/50 平均；它是數值最好的 CMS 組合，並未加入 MEPS 或 CDC。它的平均 AP **{cms_best:.6f}**，對原始 CCS HGB＋Raw CatBoost **{best:.6f}**，差 **{cms_best-best:+.6f}**。

子群分析是在既有五折上探索；各子群的 Precision／Recall／F1 重新以該子群內最高 10% 評估，不能與全體門檻數字直接替換。正例率及樣本數列於下表：

{subgroup_table}

## SHAP 解讀

Raw ICD-9 CatBoost 的前五個 Mean Absolute SHAP（log-odds 尺度）：

{raw_shap_table}

CMS CatBoost 中兩個外部特徵：

{cms_shap_table}

SHAP 只說明模型預測如何使用特徵，不等於移除特徵後的 AP 變化。`number_inpatient` 是否必要，以 E1 的重新訓練消融與配對 CI 為準；CMS 外部特徵排名低，且全體 AP 未改善。

## External Data：觀察、可能原因與未驗證假設

| 來源 | 使用的資訊與對齊方式 | 主要限制 |
| --- | --- | --- |
| CDC BRFSS | 健康問卷中的慢性病、自評健康及 DiffWalk 訊號，經共通人口／共病欄位對齊 UCI 臨床特徵；同架構的 Clinical 與 Clinical＋CDC 比較。 | 問卷測量與住院 EHR 不同；部分特徵只能映射到疾病群組，缺乏患者實測值。 |
| MEPS | 年度利用率、30 天事件及其 ICD-9／CCS 疾病群組統計，按診斷群組映射到 UCI；共同訓練另以 UCI-only NN 配對。 | 年度標籤與 30 天再住院不等價；事件群組樣本較少，與 UCI 既有利用史可能重疊。 |
| CMS HRRP | 四種疾病的全國彙總 30 天再住院率，以 UCI 主要診斷 CCS 映射；未命中者保留缺失／對應旗標。 | 只覆蓋 16.5%，沒有個人級資料，疾病類別有限；本次 CMS 指標期間為 2023–2025，與歷史 UCI 住院期間不同。 |

- **CDC：已觀察** Clinical 基線 AP **0.234919**，加入 CDC 健康訊號 **0.235256**（差 **{cdc_delta:+.6f}**），DiffWalk 真值與置換對照接近。**可能原因**是問卷的自評健康與 UCI 住院紀錄測量方式不同，且映射的共病旗標不等於真實病史。這些是 domain shift 與個人化不足的可能解釋，**不是已證明的因果原因**。CDC 沒有再住院標籤。
- **MEPS：已觀察** CCS HGB＋Raw CatBoost AP **0.247374**，換為 MEPS HGB＋Raw CatBoost **0.247139**；共同訓練 NN 從 **0.216620** 到 **0.219264**（差 **{meps_delta:+.6f}**，原配對 CI 見表）。**可能原因**是年度利用率與 30 天標籤不完全相同，以及外部利用訊號與 `number_inpatient` 重疊；這些尚未被正式因果分解。共同訓練的小幅正差支持 MEPS 在該神經網路設定下帶來可用訊號，但未使目前最佳組合改善。
- **CMS：已觀察** 四類彙總率覆蓋 16.5%；最佳 CMS 組合 AP **{cms_best:.6f}** 低於基準 **{best:.6f}**。CMS HGB＋原 CatBoost 的配對 CI 完全小於零。**可能原因**是來源只有四類群組、沒有個人狀態，且疾病訊息與原有 ICD-9／CCS 重疊。是否在某些年齡或疾病子群較有用仍屬未驗證假設；本次子群表僅供探索。

## 可以直接放入 Final Presentation 的文字

> 在相同的 99,343 次就醫、69,990 位病人、seed 777 病人互斥五折下，Raw ICD-9 CatBoost 的平均 PR-AUC (Average Precision, AP) 為 0.246506；與 CCS HGB 固定 50/50 融合達 **{best:.6f}**。不含 CCS 的 Raw HGB＋Raw CatBoost 融合為 **{raw_ensemble:.6f}**。CDC、MEPS 與 CMS 均採真實外部資料；在各自同架構配對比較中，尚未確認它們能提升最高的 HGB＋CatBoost 基準。MEPS 30 天共同訓練對同架構神經網路有小幅正差；這與最佳融合模型的比較是不同問題。

## 尚缺或不適用

{missing_text}

## 重現命令

```bash
cd {ROOT}
python scripts/experiments/run_final_catboost_ablation.py --ablations E1_no_number_inpatient E2_no_raw_diagnoses E3_no_all_diagnosis_derived --threads 8
python scripts/experiments/complete_final_presentation.py --bootstrap {args.bootstrap}
```

模型設定：CatBoost CPU、原有 46 個出院時特徵、`depth` 4／6、`iterations` 250／500／750、learning rate 0.035、L2 5／15；內層每個 outer train 用 20% 病人互斥驗證，按 AP 選參，tie-break Brier；最終全 outer train 重訓。詳細設定見 `catboost_ablation_run.json`，原始模型設定見 `results/paper_fair_comparison_seed777/run.json`。
"""
    (args.outdir / "FINAL_EXPERIMENT_SUMMARY.md").write_text(full, encoding="utf-8")


def run(args) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    sources, reference, audit = load_sources()
    scores = reference.copy()
    model_sources = {}
    for name, source, column, category in MODEL_SPECS:
        score = pd.to_numeric(sources[source][column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError(f"Invalid model probabilities: {name}")
        scores[name] = score
        model_sources[name] = category
    scores["raw_hgb_catboost_raw_50"] = (scores.roy_hgb_46 + scores.catboost_raw) / 2
    model_sources["raw_hgb_catboost_raw_50"] = "CCS ensemble ablation"
    if not np.allclose(scores.ccs_only_catboost_raw_50,
                       (scores.ccs_only_hgb + scores.catboost_raw) / 2, atol=1e-12):
        raise ValueError("Saved CCS ensemble is not the 50/50 probability average")
    if not np.allclose(scores.ccs_meps_catboost_raw_50,
                       (scores.ccs_meps_group_hgb + scores.catboost_raw) / 2, atol=1e-12):
        raise ValueError("Saved MEPS ensemble is not the 50/50 probability average")
    if not np.allclose(scores.ccs_only_hgb_cms_catboost_50,
                       (scores.ccs_only_hgb + scores.cms_catboost_raw) / 2, atol=1e-12):
        raise ValueError("Saved best CMS ensemble is not the 50/50 probability average")
    ablation_path = args.outdir / "catboost_feature_ablation_oof.csv"
    ablated = []
    if ablation_path.exists():
        complete = pd.read_csv(ablation_path)
        if len(complete) != len(reference) or complete.encounter_id.duplicated().any():
            raise ValueError("Ablation OOF cohort differs")
        complete = complete.set_index("encounter_id").loc[reference.encounter_id].reset_index()
        if not complete[KEYS].equals(reference):
            raise ValueError("Ablation OOF keys differ")
        for name in ("E1_no_number_inpatient", "E2_no_raw_diagnoses", "E3_no_all_diagnosis_derived"):
            if name in complete and complete[name].notna().all():
                scores[name] = complete[name].to_numpy(dtype=float)
                model_sources[name] = "CatBoost ablation"
                ablated.append(name)
    metrics, folds = summarize(scores, model_sources)
    for name, expected in BASELINE_AP.items():
        actual = metrics.set_index("Model").loc[name, "PR-AUC"]
        if not np.isclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError(f"Original baseline changed: {name}, {actual} vs {expected}")
    metrics.to_csv(args.outdir / "final_experiment_metrics.csv", index=False)
    folds.to_csv(args.outdir / "final_fold_metrics.csv", index=False)
    ablation = metrics.loc[metrics.Model.isin(["catboost_raw", *ablated]), ["Model", *METRIC_COLUMNS,
                                                                           "Pooled_OOF_AP", "Folds"]].copy()
    ablation["Full_minus_Ablated_PR_AUC"] = (
        metrics.set_index("Model").loc["catboost_raw", "PR-AUC"] - ablation["PR-AUC"])
    ablation["Model"] = ablation.Model.replace({"catboost_raw": "E0_full_catboost"})
    ablation.to_csv(args.outdir / "catboost_feature_ablation.csv", index=False)
    ablation.to_csv(args.outdir / "final_ablation_metrics.csv", index=False)
    selections = []
    for name in ablated:
        for fold in range(1, 6):
            choice = args.outdir / f"{name}_fold_{fold}_selection.json"
            if not choice.exists():
                raise ValueError(f"Missing CatBoost selection record: {choice}")
            selections.append({"Ablation": name, **json.loads(choice.read_text())})
    pd.DataFrame(selections, columns=["Ablation", "Fold", "Depth", "Trees", "Validation_PR_AUC",
                                      "Validation_Brier", "Validation_Patients"]).to_csv(
        args.outdir / "catboost_ablation_selections.csv", index=False)
    metrics.loc[metrics.Model.isin(["raw_hgb_catboost_raw_50", "ccs_only_catboost_raw_50"])].to_csv(
        args.outdir / "ccs_ensemble_ablation.csv", index=False
    )
    coverage = cms_coverage(args, scores)
    coverage.to_csv(args.outdir / "cms_coverage_analysis.csv", index=False)
    metrics.loc[metrics.Model.isin(["ccs_only_hgb", "catboost_raw", "ccs_only_catboost_raw_50",
                                   "cms_hgb", "cms_catboost_raw", "cms_hgb_catboost_raw_50",
                                   "ccs_only_hgb_cms_catboost_50", "cms_hgb_cms_catboost_50"])].to_csv(
        args.outdir / "cms_complete_metrics.csv", index=False
    )
    pd.DataFrame(audit).to_csv(args.outdir / "source_audit.csv", index=False)

    cms_pairs = [("cms_hgb", "ccs_only_hgb"), ("cms_catboost_raw", "catboost_raw"),
                 ("cms_hgb_catboost_raw_50", "ccs_only_catboost_raw_50"),
                 ("ccs_only_hgb_cms_catboost_50", "ccs_only_catboost_raw_50"),
                 ("cms_hgb_cms_catboost_50", "ccs_only_catboost_raw_50")]
    cms_bootstrap = existing_bootstrap(
        scores, ROOT / "results/cms_group_prior_5fold_seed777/paired_ap_bootstrap.csv", cms_pairs, "CMS saved"
    )
    cms_bootstrap.to_csv(args.outdir / "cms_paired_bootstrap.csv", index=False)
    copied = [cms_bootstrap]
    copied.append(existing_bootstrap(scores, ROOT / "results/paper_fair_comparison_seed777/paired_ap_bootstrap.csv",
                                     [("ccs_meps_catboost_raw_50", "ccs_only_catboost_raw_50")], "MEPS saved"))
    copied.append(existing_bootstrap(scores, ROOT / "results/meps_30d_joint_transfer_seed777/paired_ap_bootstrap.csv",
                                     [("joint_meps_30d", "uci_only_nn")], "MEPS joint saved"))
    new_pairs = [("ccs_only_catboost_raw_50", "raw_hgb_catboost_raw_50"),
                 ("E4_Clinical_CDCHealth", "E2_Clinical"),
                 ("Only_DiffWalk", "Permuted_DiffWalk")]
    new_pairs += [("catboost_raw", name) for name in ablated]
    if args.bootstrap:
        paired_patient_bootstrap(scores, new_pairs, args.outdir, args.bootstrap, 777)
        fresh = pd.read_csv(args.outdir / "paired_ap_bootstrap.csv")
        fresh["Source"] = "recomputed from audited OOF"
        fresh["Reused"] = False
        copied.append(fresh)
        fresh.loc[fresh.Treatment.eq("ccs_only_catboost_raw_50")].to_csv(
            args.outdir / "ccs_ensemble_bootstrap.csv", index=False
        )
        fresh.loc[fresh.Treatment.eq("catboost_raw")].to_csv(
            args.outdir / "catboost_feature_ablation_bootstrap.csv", index=False
        )
    bootstrap = pd.concat(copied, ignore_index=True)
    bootstrap.to_csv(args.outdir / "final_bootstrap_comparisons.csv", index=False)
    source_pairs = [
        ("CDC BRFSS", "E2_Clinical", "E4_Clinical_CDCHealth", "問卷健康訊號映射到 UCI 臨床群組"),
        ("CDC BRFSS", "Permuted_DiffWalk", "Only_DiffWalk", "DiffWalk 真值對置換控制"),
        ("MEPS", "ccs_only_catboost_raw_50", "ccs_meps_catboost_raw_50", "ICD-9／CCS 疾病群組風險對齊"),
        ("MEPS", "uci_only_nn", "joint_meps_30d", "病人互斥的 MEPS 30 天事件共同訓練"),
        ("CMS HRRP", "ccs_only_hgb", "cms_hgb", "主要診斷 CCS 對應四類全國彙總率"),
        ("CMS HRRP", "catboost_raw", "cms_catboost_raw", "主要診斷 CCS 對應四類全國彙總率"),
        ("CMS HRRP", "ccs_only_catboost_raw_50", "cms_hgb_catboost_raw_50", "CMS HGB 與原始 CatBoost 融合"),
        ("CMS HRRP", "ccs_only_catboost_raw_50", "ccs_only_hgb_cms_catboost_50", "原始 CCS HGB 與 CMS CatBoost 融合"),
    ]
    by_model = metrics.set_index("Model")
    external_rows = []
    for source, control, treatment, alignment in source_pairs:
        matched = bootstrap.loc[bootstrap.Treatment.eq(treatment) & bootstrap.Control.eq(control)]
        row = {"External_Source": source, "Control": control, "Treatment": treatment,
               "Alignment": alignment, "Delta_PR_AUC": delta(scores, treatment, control),
               "CI95_Lower": float(matched.CI95_Lower.iloc[0]) if len(matched) else np.nan,
               "CI95_Upper": float(matched.CI95_Upper.iloc[0]) if len(matched) else np.nan}
        for metric in METRIC_COLUMNS:
            row[f"Control_{metric}"] = by_model.loc[control, metric]
            row[f"Treatment_{metric}"] = by_model.loc[treatment, metric]
        external_rows.append(row)
    pd.DataFrame(external_rows).to_csv(args.outdir / "external_data_comparisons.csv", index=False)
    missing = []
    for name in ("E1_no_number_inpatient", "E2_no_raw_diagnoses", "E3_no_all_diagnosis_derived"):
        if name not in ablated:
            missing.append({"Item": name, "Status": "尚未完成", "Reason": "需要五折重新訓練 CatBoost；原始 OOF 無此預測"})
    if not args.bootstrap:
        missing.append({"Item": "new paired bootstrap", "Status": "尚未完成", "Reason": "本次 --bootstrap 0，只有已存 CI"})
    missing_table = pd.DataFrame(missing, columns=["Item", "Status", "Reason"])
    missing_table.to_csv(args.outdir / "final_missing_results.csv", index=False)
    audit_json = {"rows": len(reference), "patients": int(reference.patient_nbr.nunique()),
                  "seed": 777, "patient_disjoint": True, "cms_mapped_rows": int(coverage.loc[
                      coverage.Subset.eq("CMS-covered"), "N"].iloc[0]),
                  "cms_mapped_fraction": float(coverage.loc[
                      coverage.Subset.eq("CMS-covered"), "N"].iloc[0] / len(reference)),
                  "metric_policy": "mean of five outer folds, top 10% test scores per fold",
                  "files": audit}
    (args.outdir / "audit.json").write_text(json.dumps(audit_json, indent=2, ensure_ascii=False) + "\n")
    write_report(args, metrics, ablation, bootstrap, coverage, missing_table, audit_json)
    print(metrics[["Model", *METRIC_COLUMNS]].round(6).to_string(index=False))
    print(f"Saved final presentation artifacts to {args.outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--cms-prior", type=Path, default=ROOT / "data/processed/cms/cms_readmission_ccs_prior.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/final_presentation_completion")
    parser.add_argument("--bootstrap", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
