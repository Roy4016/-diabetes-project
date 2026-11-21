# 論文復現的同切分比較

## 已核對的資料與結果

本專題最新 `results/meps_event_ccs_group_hgb_seed777/oof_all.csv` 與
`Bhuvan_Rajpoot_SSRN/5-fold-reproduce/results/five_fold/baseline/oof_predictions.csv.gz`
逐筆核對了 `encounter_id`、`patient_nbr`、30 天再入院標籤與 `Fold`：**99,343 筆全部相同**，
共 69,990 位病人，五個測試折分別有 19,869、19,869、19,868、19,869、19,868 筆。
UCI 原始 CSV SHA-256 是
`0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`。

`preprocess.py` 目前的輸出另排除了 3 筆 `gender=Unknown/Invalid`，所以是 **99,340 筆**；
它的折別也不同。此比較以最新 HGB／MEPS 實驗的 99,343 筆 OOF 折別為唯一基準，
不讀取 `preprocess.py` 的舊切分。

| 現有同切分方法 | 五折平均 PR-AUC／AP | 五折平均 AUROC |
|---|---:|---:|
| Rajpoot 復現 HGB | 0.228703 | 0.671776 |
| 同復現中的 46 欄 one-hot HGB | 0.237261 | 0.676968 |
| 本專題 CCS-only HGB | 0.239179 | 0.678333 |
| 本專題 MEPS event-CCS HGB | 0.239077 | 0.677884 |

`Bhuvan_Rajpoot_SSRN/hgb_catboost/` 的 CatBoost 0.251383 AP、
HGB＋CatBoost 0.249442 AP（擴充特徵融合 0.251195）來自**另一個 80/20 留出測試**，
不能與上表五折平均值直接排序。`salim-ibrahim-2026-readmission` 的 `shap-metrics`
分支是 Rajpoot 復現的**指標與 SHAP 延伸分析**，沿用同一個 80/20 測試集；
該 repo 沒有可在這份五折上直接比較的獨立 Salim 模型。

## 五折補跑方式

在 `diabetes_project` 目錄執行：

```bash
python -m pip install -e '.[comparison]'

# 先確認來源與折別；不訓練模型
python scripts/experiments/run_paper_fair_comparison.py \
  --audit-only --outdir results/paper_comparison_audit_seed777

# 補跑 CatBoost 與 SHAP；預設使用 CPU、五折、兩種 ICD 表示
python scripts/experiments/run_paper_fair_comparison.py \
  --outdir results/paper_fair_comparison_seed777 \
  --threads 8 --shap-sample 1000 --bootstrap 2000
```

若先測一折的執行環境，可在完整指令中加入 `--max-folds 1 --variants family`，
並使用**另一個 outdir**；少於五折的結果僅供檢查程式，不能列入正式排行榜。

腳本會在每個外層訓練折內，用病人互斥的 20% 驗證資料選 CatBoost 深度
（4 或 6）與輪數（250、500、750），只用該折訓練資料重訓後預測外層測試折。
`catboost_family` 沿用 Rajpoot 的 46 個出院欄位和 ICD-9 章節；
`catboost_raw` 保留原始 ICD-9 碼。固定 50/50 機率融合分別對照 Rajpoot HGB 與
本專題含 MEPS 的 HGB；腳本也計算不含 MEPS 的 CCS-only HGB＋CatBoost，
以相同 CatBoost 預測直接測量外部資料的增益。`ccs_meps_refit_hgb` 用既有每折選定參數重訓，
供 SHAP 解釋；`run.json` 會記錄它與原 OOF 預測的最大差異。開發時第 1 折重訓
的 19,869 筆機率與既有 OOF 完全一致（最大絕對差 0）。

若訓練與 SHAP 已完成，但最後整理報表時中斷，可用現有的 `oof_all.csv.gz`
重新產生指標與 `run.json`，不再訓練模型：

```bash
python scripts/experiments/run_paper_fair_comparison.py \
  --report-only --bootstrap 2000 \
  --outdir results/paper_fair_comparison_seed777
```

`fold_metrics.csv` 與 `summary.csv` 同時輸出 AUROC、PR-AUC（`average_precision_score`）、
Precision、Recall、F1-score 與 Brier Score。Precision／Recall／F1 採每折**預先固定的
最高風險 10% 警示量**，只依分數排序，不使用測試標籤選門檻；機率指標與此門檻無關。
`Pooled_PR_AUC` 另列，不與五折平均 AP 混用。
`paired_ap_bootstrap.csv` 對同一批病人在每折內配對重抽，提供五折平均 AP 差值與
95% 區間；預設 500 次，可用 `--bootstrap` 調整。

`shap_global.csv`、`shap_fold.csv` 與 `shap_*.png` 顯示 CatBoost 及本專題
MEPS HGB 的特徵影響排名。SHAP 在外層測試折抽樣 1,000 筆計算，
以 log-odds 尺度的平均絕對 SHAP 值排序；設 `--shap-sample 0` 可解釋全部測試筆數。
HGB 的 one-hot 欄位先逐筆加總回原始欄位再排名。SHAP 描述模型預測的歸因，
不代表臨床因果關係。

原始碼來源：Rajpoot 復現 `7c40041`、Salim SHAP 分支 `8b03174`。
本專題的 46 欄重編碼根據 Rajpoot 復現 MIT 授權程式調整；詳細來源見
`src/diabetes_project/paper_comparison.py`。本比較**沒有重現 Salim & Ibrahim 原論文模型**。

## 2026-11-21 五折補跑結果

已完成五折 CatBoost 訓練及每折 1,000 筆測試資料的 SHAP；本專題 MEPS HGB
每折重訓機率與原 OOF 完全一致。下表數值是**五折 PR-AUC 平均**，
與單一測試集的 pooled AP 分開報告：

| 方法 | 五折平均 PR-AUC | pooled AP | 五折平均 AUROC | Brier |
|---|---:|---:|---:|---:|
| 本專題 CCS-only HGB | 0.239179 | 0.237823 | 0.678333 | 0.095470 |
| 本專題 MEPS HGB | 0.239077 | 0.238043 | 0.677884 | 0.095500 |
| CatBoost 診斷家族 | 0.244454 | 0.243281 | 0.677482 | 0.095233 |
| CatBoost 原始 ICD-9 | 0.246506 | 0.245430 | 0.682839 | 0.095084 |
| CCS-only HGB＋CatBoost 診斷家族 | 0.247337 | **0.246005** | 0.681428 | 0.095093 |
| MEPS HGB＋CatBoost 診斷家族 | 0.247140 | 0.245854 | 0.681210 | 0.095118 |
| CCS-only HGB＋CatBoost 原始 ICD-9 | **0.247374** | 0.245932 | **0.683446** | 0.095068 |
| MEPS HGB＋CatBoost 原始 ICD-9 | 0.247139 | 0.245889 | 0.683233 | 0.095087 |

CatBoost 原始 ICD-9 相對 CCS-only HGB 的五折平均 AP 差為 +0.007327，
配對病人 bootstrap 95% CI 為 [0.004039, 0.010484]。**目前的 MEPS 融合
未顯示比相同 CatBoost 下的 CCS-only 融合更好**：診斷家族版差值 −0.000197
（95% CI [−0.001026, 0.000604]），原始 ICD-9 版差值 −0.000234
（95% CI [−0.001096, 0.000547]）；完整對照見
`results/paper_fair_comparison_seed777/summary.csv`。最高 PR-AUC 0.247374
比文獻數字 0.237 高 0.010374，但兩者測試協定不同，不能宣稱已在同一測試集
超越該論文。

全域 SHAP 以測試折平均絕對 log-odds 貢獻排序：三個模型第一名皆為
`number_inpatient`，第二名皆為 `discharge_disposition_id`。MEPS HGB 的
`meps_ccs_revisit_mean` 排第五（0.040843），代表模型使用了外部訊號；
它的存在本身不等於 PR-AUC 有增益。
