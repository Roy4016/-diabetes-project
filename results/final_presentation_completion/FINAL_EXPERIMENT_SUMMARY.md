# Final Presentation 實驗統整：30 天糖尿病再住院

產生方式：`python scripts/experiments/complete_final_presentation.py --bootstrap 2000`。本檔只依據已核對的實際 OOF 和原始資料生成；不使用早期單次 holdout、99,340 筆隊列或其他 seed。

## 研究協定與可重用資料

UCI 共 **99,343** 次就醫、**69,990** 位病人，seed **777**，病人互斥五折。所有納入的 OOF 均逐筆核對 Encounter ID、Patient ID、Fold、Label。AP 為 `average_precision_score`；表列六項指標為各測試折的**算術平均**。Precision、Recall、F1 固定取每折預測風險最高 10%。合併所有 OOF 計算的 pooled AP 存在 CSV 的另一欄，不與五折平均混淆。對照資料及 SHA256 見 `source_audit.csv`。

CCS 是 ICD-9 疾病分類表，單獨使用 CCS 不算加入患者級外部資料；正式的外部資料來源限定為 CDC BRFSS、MEPS、CMS HRRP。六項指標的方向為 PR-AUC、AUROC、Precision、Recall、F1 越高越好，Brier 越低越好。

先前論文報告的 0.237 AP 使用不同的單次 holdout；它只能作背景參考。公平比較以此處同一 99,343 筆、同五折 OOF 的 Rajpoot HGB 復現（AP 0.228703）及其他同切分模型為準。

已重用：論文復現 HGB、Raw HGB、CCS HGB、Raw CatBoost、MEPS、CDC、CMS 的既有五折 OOF；CMS 既有 2,000 次病人分折 bootstrap；MEPS 既有配對 bootstrap。新計算：Raw HGB＋Raw CatBoost 固定 50/50 融合、CMS covered 子群與缺少的配對 bootstrap。Ablation 只有重新訓練後才納入。

## 主要 Baseline 與 CCS 融合

| Model | PR-AUC | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bhuvan_hgb | 0.228703 | 0.671776 | 0.272004 | 0.238729 | 0.254282 | 0.096042 |
| roy_hgb_46 | 0.237261 | 0.676968 | 0.279355 | 0.245182 | 0.261155 | 0.095577 |
| ccs_only_hgb | 0.239179 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 |
| catboost_raw | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 |
| raw_hgb_catboost_raw_50 | 0.246225 | 0.682744 | 0.287412 | 0.252253 | 0.268687 | 0.095119 |
| ccs_only_catboost_raw_50 | 0.247374 | 0.683446 | 0.288620 | 0.253313 | 0.269817 | 0.095068 |

## CMS HRRP

| Model | PR-AUC | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cms_hgb | 0.238496 | 0.678059 | 0.281873 | 0.247390 | 0.263508 | 0.095499 |
| cms_catboost_raw | 0.246507 | 0.682518 | 0.286103 | 0.251103 | 0.267463 | 0.095083 |
| cms_hgb_catboost_raw_50 | 0.246741 | 0.683289 | 0.287714 | 0.252517 | 0.268969 | 0.095083 |
| ccs_only_hgb_cms_catboost_50 | 0.247074 | 0.683315 | 0.289225 | 0.253844 | 0.270382 | 0.095072 |
| cms_hgb_cms_catboost_50 | 0.246644 | 0.683144 | 0.290232 | 0.254727 | 0.271323 | 0.095088 |

## CDC BRFSS

| Model | PR-AUC | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2_Clinical | 0.234919 | 0.678280 | 0.276334 | 0.242530 | 0.258331 | 0.095664 |
| E4_Clinical_CDCHealth | 0.235256 | 0.678403 | 0.276939 | 0.243060 | 0.258896 | 0.095640 |
| Only_DiffWalk | 0.235084 | 0.678039 | 0.277039 | 0.243148 | 0.258990 | 0.095663 |
| No_DiffWalk | 0.234110 | 0.677909 | 0.274824 | 0.241204 | 0.256919 | 0.095693 |
| Permuted_DiffWalk | 0.235028 | 0.678541 | 0.274622 | 0.241026 | 0.256730 | 0.095662 |

## MEPS

| Model | PR-AUC | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ccs_meps_group_hgb | 0.239077 | 0.677884 | 0.282276 | 0.247745 | 0.263885 | 0.095500 |
| ccs_meps_catboost_raw_50 | 0.247139 | 0.683233 | 0.286103 | 0.251104 | 0.267463 | 0.095087 |
| uci_only_nn | 0.216620 | 0.664481 | 0.263243 | 0.231040 | 0.246093 | 0.097245 |
| joint_meps_30d | 0.219264 | 0.665757 | 0.263746 | 0.231482 | 0.246563 | 0.097049 |
| joint_meps_30d_catboost50 | 0.239537 | 0.680805 | 0.283686 | 0.248983 | 0.265204 | 0.095406 |

## CatBoost 原始特徵消融

| Model | PR-AUC | AUROC | Precision | Recall | F1 | Brier | Full_minus_Ablated_PR_AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0_full_catboost | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 | 0.000000 |
| E1_no_number_inpatient | 0.220537 | 0.658885 | 0.251158 | 0.220433 | 0.234794 | 0.096609 | 0.025970 |
| E2_no_raw_diagnoses | 0.238047 | 0.673475 | 0.280060 | 0.245800 | 0.261814 | 0.095611 | 0.008460 |
| E3_no_all_diagnosis_derived | 0.236696 | 0.671660 | 0.277845 | 0.243855 | 0.259743 | 0.095702 | 0.009811 |

E0 是原始 46 欄 Raw ICD-9 CatBoost，直接重用 OOF。E1 移除 `number_inpatient`；E2 只移除 `diag_1`、`diag_2`、`diag_3`；E3 另外移除由診斷資料計算的 `number_diagnoses`。此 Raw CatBoost 未含 CCS 或其他診斷衍生欄位。每組已完成的消融均沿用原本的病人互斥內層選參程序並重新訓練；未完成者只列於缺項表。

## 配對比較與不確定性

以下差值均為五折 AP 差的平均。Bootstrap 單位為每折內的病人；95% CI 不能直接解讀為來源造成改變的因果證據。完整比較見 `final_bootstrap_comparisons.csv`。

| Treatment | Control | MeanFold_Delta_PR_AUC | CI95_Lower | CI95_Upper |
| --- | ---: | ---: | ---: | ---: |
| cms_hgb_catboost_raw_50 | ccs_only_catboost_raw_50 | -0.000633 | -0.001170 | -0.000063 |
| ccs_only_hgb_cms_catboost_50 | ccs_only_catboost_raw_50 | -0.000300 | -0.000908 | 0.000487 |
| joint_meps_30d | uci_only_nn | 0.002644 | 0.000035 | 0.005273 |
| ccs_only_catboost_raw_50 | raw_hgb_catboost_raw_50 | 0.001148 | 0.000319 | 0.002043 |
| E4_Clinical_CDCHealth | E2_Clinical | 0.000337 | -0.000653 | 0.001471 |
| catboost_raw | E1_no_number_inpatient | 0.025970 | 0.020712 | 0.030794 |
| catboost_raw | E2_no_raw_diagnoses | 0.008460 | 0.005983 | 0.010851 |
| catboost_raw | E3_no_all_diagnosis_derived | 0.009811 | 0.007082 | 0.012389 |

## CMS coverage 與特徵內容

CMS 只依 UCI 主要診斷 CCS 對應四個概念：心肌梗塞、心衰竭、肺炎、COPD。命中率為 **16,413/99,343（16.5%）**。數值是 CMS 全國風險標準化再住院率，並非個人標籤。`ccs_only_hgb_cms_catboost_50` 是原始 **46 欄＋CCS 的 HGB**，與 **原始 46 欄＋CMS rate／mapped indicator 的 CatBoost** 以預測機率 50/50 平均；它是數值最好的 CMS 組合，並未加入 MEPS 或 CDC。它的平均 AP **0.247074**，對原始 CCS HGB＋Raw CatBoost **0.247374**，差 **-0.000300**。

子群分析是在既有五折上探索；各子群的 Precision／Recall／F1 重新以該子群內最高 10% 評估，不能與全體門檻數字直接替換。正例率及樣本數列於下表：

| Subset | Model | N | Positive_Count | Positive_Rate | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMS-covered | ccs_only_catboost_raw_50 | 16413 | 2011 | 0.122525 | 0.225494 |
| CMS-covered | ccs_only_hgb_cms_catboost_50 | 16413 | 2011 | 0.122525 | 0.226265 |
| CMS-uncovered | ccs_only_catboost_raw_50 | 82930 | 9303 | 0.112179 | 0.251922 |
| CMS-uncovered | ccs_only_hgb_cms_catboost_50 | 82930 | 9303 | 0.112179 | 0.251448 |

## SHAP 解讀

Raw ICD-9 CatBoost 的前五個 Mean Absolute SHAP（log-odds 尺度）：

| Feature | MeanAbsSHAP_logodds | Rank |
| --- | ---: | ---: |
| number_inpatient | 0.238927 | 1 |
| discharge_disposition_id | 0.139130 | 2 |
| diag_1 | 0.077924 | 3 |
| diag_3 | 0.059604 | 4 |
| payer_code | 0.042366 | 5 |

CMS CatBoost 中兩個外部特徵：

| Feature | MeanAbsSHAP_logodds | Rank |
| --- | ---: | ---: |
| cms_readmission_rate | 0.006701 | 23 |
| cms_mapped | 0.000349 | 39 |

SHAP 只說明模型預測如何使用特徵，不等於移除特徵後的 AP 變化。`number_inpatient` 是否必要，以 E1 的重新訓練消融與配對 CI 為準；CMS 外部特徵排名低，且全體 AP 未改善。

## External Data：觀察、可能原因與未驗證假設

| 來源 | 使用的資訊與對齊方式 | 主要限制 |
| --- | --- | --- |
| CDC BRFSS | 健康問卷中的慢性病、自評健康及 DiffWalk 訊號，經共通人口／共病欄位對齊 UCI 臨床特徵；同架構的 Clinical 與 Clinical＋CDC 比較。 | 問卷測量與住院 EHR 不同；部分特徵只能映射到疾病群組，缺乏患者實測值。 |
| MEPS | 年度利用率、30 天事件及其 ICD-9／CCS 疾病群組統計，按診斷群組映射到 UCI；共同訓練另以 UCI-only NN 配對。 | 年度標籤與 30 天再住院不等價；事件群組樣本較少，與 UCI 既有利用史可能重疊。 |
| CMS HRRP | 四種疾病的全國彙總 30 天再住院率，以 UCI 主要診斷 CCS 映射；未命中者保留缺失／對應旗標。 | 只覆蓋 16.5%，沒有個人級資料，疾病類別有限；本次 CMS 指標期間為 2023–2025，與歷史 UCI 住院期間不同。 |

- **CDC：已觀察** Clinical 基線 AP **0.234919**，加入 CDC 健康訊號 **0.235256**（差 **+0.000337**），DiffWalk 真值與置換對照接近。**可能原因**是問卷的自評健康與 UCI 住院紀錄測量方式不同，且映射的共病旗標不等於真實病史。這些是 domain shift 與個人化不足的可能解釋，**不是已證明的因果原因**。CDC 沒有再住院標籤。
- **MEPS：已觀察** CCS HGB＋Raw CatBoost AP **0.247374**，換為 MEPS HGB＋Raw CatBoost **0.247139**；共同訓練 NN 從 **0.216620** 到 **0.219264**（差 **+0.002644**，原配對 CI 見表）。**可能原因**是年度利用率與 30 天標籤不完全相同，以及外部利用訊號與 `number_inpatient` 重疊；這些尚未被正式因果分解。共同訓練的小幅正差支持 MEPS 在該神經網路設定下帶來可用訊號，但未使目前最佳組合改善。
- **CMS：已觀察** 四類彙總率覆蓋 16.5%；最佳 CMS 組合 AP **0.247074** 低於基準 **0.247374**。CMS HGB＋原 CatBoost 的配對 CI 完全小於零。**可能原因**是來源只有四類群組、沒有個人狀態，且疾病訊息與原有 ICD-9／CCS 重疊。是否在某些年齡或疾病子群較有用仍屬未驗證假設；本次子群表僅供探索。

## 可以直接放入 Final Presentation 的文字

> 在相同的 99,343 次就醫、69,990 位病人、seed 777 病人互斥五折下，Raw ICD-9 CatBoost 的平均 PR-AUC 為 0.246506；與 CCS HGB 固定 50/50 融合達 **0.247374**。不含 CCS 的 Raw HGB＋Raw CatBoost 融合為 **0.246225**。CDC、MEPS 與 CMS 均採真實外部資料；在各自同架構配對比較中，尚未確認它們能提升最高的 HGB＋CatBoost 基準。MEPS 30 天共同訓練對同架構神經網路有小幅正差；這與最佳融合模型的比較是不同問題。

## 尚缺或不適用

無。

## 重現命令

```bash
cd /media/synology/G/314554044/diabetes_project
python scripts/experiments/run_final_catboost_ablation.py --ablations E1_no_number_inpatient E2_no_raw_diagnoses E3_no_all_diagnosis_derived --threads 8
python scripts/experiments/complete_final_presentation.py --bootstrap 2000
```

模型設定：CatBoost CPU、原有 46 個出院時特徵、`depth` 4／6、`iterations` 250／500／750、learning rate 0.035、L2 5／15；內層每個 outer train 用 20% 病人互斥驗證，按 AP 選參，tie-break Brier；最終全 outer train 重訓。詳細設定見 `catboost_ablation_run.json`，原始模型設定見 `results/paper_fair_comparison_seed777/run.json`。
