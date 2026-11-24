# MEPS＋NHANES 雙外部資料：群組統計與多任務訓練

本實驗固定本專題最新 **99,343 筆、69,990 位病人、seed 777、病人互斥五折**，使用 [同一份 OOF](../results/meps_event_ccs_group_hgb_seed777/oof_all.csv) 與已選定的 CCS-only HGB 參數。兩份外部來源均為真實資料：MEPS 有 **21,176 個相鄰年度配對**，NHANES 有 **1,747 位自述糖尿病受訪者**。另讀取既有原始 ICD-9 CatBoost 五折 OOF，在每種 HGB 上做**固定 50/50** 機率平均，與目前最佳流程比較。

## 為什麼分開處理標籤

MEPS 外部任務是「下一年是否有急診」與「下一年是否住院」，NHANES 是「過去一年是否曾住院」，UCI 主任務是「本次出院後 30 天內是否再住院」。三者時間窗和事件定義不同；本實驗**不把標籤直接合併成同一個 30 天目標**。外部資料只用於產生特徵，HGB 只使用每個外層訓練折的 UCI 30 天標籤訓練。

## 兩種接法

### 1. Grouping：分層群組統計

將年齡分為 20 歲區間，結合性別與共同種族類別。MEPS 的次年 ER、次年住院和 NHANES 的過去一年住院**各自**計算來源內的 survey-weighted 比例；權重先按同一人的重複紀錄數調整、正規化，再截在 0.1–10。接著用年齡→年齡＋性別→年齡＋性別＋種族的階層平滑（預設強度 50）避免稀疏群組不穩定。輸出每個來源的平滑率、相對其來源整體盛行率的 log-odds lift、群組支援量；雙來源版再加入兩個住院 lift 的平均與差距。平均的是**來源內的相對偏移**，沒有將兩種原始標籤或原始住院率當成相同事件池化。

### 2. Multitask：共用編碼器、來源專屬預測頭

只使用兩個來源都可對齊的年齡、性別及種族欄位，訓練一個小型 PyTorch 共用編碼器。MEPS 有 ER、住院兩個 head；NHANES 有住院 head。每步分別計算來源內的 weighted binary cross-entropy，再平均來源損失，避免 21,176 筆 MEPS 使 1,747 筆 NHANES 失去權重。來源訓練**不接觸 UCI 標籤**；將各 head 的風險分數與編碼器表示轉成 UCI HGB 的外部特徵。MEPS-only、NHANES-only、dual 三種版本各自訓練，供消融比較。

Grouping 和 Multitask 的 MEPS-only／NHANES-only／dual 版本都在每個外層訓練折用**同一組 CCS-only HGB 參數**訓練；該參數原本從該折內層驗證選出。資料編碼器只在外層 UCI 訓練折擬合。外部特徵模型僅使用獨立的 MEPS／NHANES 資料，沒有看外層 UCI 測試標籤。腳本會逐筆核對 CatBoost OOF 的 encounter、patient、label 和 fold，並在第 1 折重訓 CCS-only HGB，比對既有 OOF 機率。

## 執行

```bash
cd /media/synology/G/314554044/diabetes_project
python -m pip install -e '.[comparison,dual_external]'

# 完整五折；兩種方法、三種外部來源版本及固定 50/50 CatBoost 融合
python scripts/experiments/run_dual_external_meps_nhanes.py \
  --outdir results/meps_nhanes_dual_external_seed777 \
  --epochs 30 --bootstrap 500 --torch-threads 4
```

短跑可加 `--max-folds 1 --epochs 2 --bootstrap 0`，並使用另一個 `--outdir`。**單折不是完整五折結果**。預設 CPU 訓練外部編碼器，避免占用忙碌中的 GPU。

輸出：`external_sources.csv` 紀錄外部資料規模與陽性率；`external_training.csv` 紀錄多任務訓練 loss；`oof_all.csv.gz` 為逐筆折外預測；`fold_metrics.csv` 與 `summary.csv` 有 **AUROC、PR-AUC (Average Precision, AP)、Precision、Recall、F1、Brier**，Precision／Recall／F1 採每折固定最高風險 10% 警示量；`paired_ap_bootstrap.csv` 是折內病人配對 AP 差值的 95% 區間；`config.json` 保存切分與標籤定義。`Pooled_PR_AUC` 另列，不與五折平均 AP 混用。

關鍵判讀是先看 `grouping_dual` 是否超過 `grouping_meps` 與 `grouping_nhanes`，再看 `multitask_dual` 是否超過相應單來源版本。最後以 `*_catboost_raw50` 對 `ccs_only_catboost_raw50`，檢查雙來源是否改善目前最佳流程。僅有雙來源模型分數高於 SOTA 文獻數值，不等於外部資料帶來額外增益；仍需同切分配對對照。

這兩份外部資料能共同對齊的欄位主要是人口特徵；NHANES 住院標籤的時間窗不同，故兩種方法可能只學到與 UCI 年齡／種族已有訊號重疊的資訊。若雙來源未改善，應如實報告，不以來源數量推斷有效性。

## 2026-11-22 五折結果

主指標為五折平均 PR-AUC (Average Precision, AP)；所有方法都用相同 99,343 筆和相同病人折別。下表是已完成的五折摘要，完整六指標、折間標準差與 pooled AP 見 [summary.csv](../results/meps_nhanes_dual_external_seed777/summary.csv)。

| 方法 | 五折平均 AP | 平均 AUROC | Brier | 說明 |
|---|---:|---:|---:|---|
| CCS-only HGB | **0.239179** | 0.678333 | **0.095470** | 不使用真實外部病人資料的 HGB 基線 |
| 原有 MEPS event-CCS HGB | 0.239077 | 0.677884 | 0.095500 | 原有外部資料接法 |
| Grouping：MEPS-only | 0.237652 | 0.678020 | 0.095536 | 來源專屬群組統計 |
| Grouping：NHANES-only | 0.237939 | 0.678102 | 0.095526 | 來源專屬群組統計 |
| **Grouping：雙來源** | **0.237757** | 0.677639 | 0.095551 | 兩個相對群組風險一同加入 |
| Multitask：MEPS-only | 0.237703 | 0.677568 | 0.095547 | 外部來源專屬預訓練 |
| Multitask：NHANES-only | 0.237839 | 0.677914 | 0.095541 | 外部來源專屬預訓練 |
| **Multitask：雙來源** | **0.238303** | 0.677964 | 0.095536 | 共用編碼器與來源專屬 head |

加入既有 CatBoost 原始 ICD-9 OOF 做固定 50/50 融合後：

| 融合方法 | 五折平均 AP | 平均 AUROC | Brier |
|---|---:|---:|---:|
| CCS-only HGB＋CatBoost | **0.247374** | **0.683446** | **0.095068** |
| 原有 MEPS HGB＋CatBoost | 0.247139 | 0.683233 | 0.095087 |
| Grouping 雙來源 HGB＋CatBoost | 0.246639 | 0.683150 | 0.095102 |
| Multitask 雙來源 HGB＋CatBoost | 0.246877 | 0.683299 | 0.095096 |

目前雙來源多任務 AP 比其 MEPS-only／NHANES-only 版本各高約 0.000600／0.000464，但兩個配對病人 bootstrap 95% CI 分別為 **[−0.001059, +0.002080]**、**[−0.000950, +0.001719]**，都跨零。相對 CCS-only HGB，多任務雙來源 ΔAP **−0.000876**、CI **[−0.002436, +0.000313]**；群組雙來源 ΔAP **−0.001422**、CI **[−0.003072, +0.000049]**。加入 CatBoost 後，最佳雙來源融合是多任務版 **0.246877**，比現有 CCS-only 融合低 **0.000496**，CI **[−0.001267, +0.000255]**；比原有 MEPS event-CCS 融合低 **0.000262**，CI **[−0.001238, +0.000658]**。完整 17 組配對比較見 [bootstrap](../results/meps_nhanes_dual_external_seed777/paired_ap_bootstrap.csv)。**結論：在目前同切分下，兩個外部來源同時加入沒有可確認的 PR-AUC (Average Precision, AP) 改善。**
