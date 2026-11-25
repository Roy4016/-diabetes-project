# 糖尿病 30 天再住院：五折實驗與完整結果分析

更新日期：2026-11-25。正式比較限於同一 99,343 筆、seed 777 病人互斥五折；CDC、MEPS、CMS 為正式 external data。下方「最終 External Data 單來源接法」以 CDC CatBoost Teacher、全疾病 MEPS Teacher、CMS HRRP 的 **1／2／2 個欄位**為準；先前 XGBoost Teacher、四欄 MEPS 統計與三來源聯合實驗保留於歷史紀錄，不與新版混稱。

原有正式主表由 `complete_final_presentation.py` 與 `build_results_analysis.py` 產生；本次另依後續實驗各自保存的五折摘要及配對分析補充章節。重新執行舊報告產生器可能覆寫新增段落，更新報告時請先保留本檔。正式比較不使用早期單次 holdout、99,340 筆隊列或其他 seed。

主要輸出：[六項指標](results/final_presentation_completion/final_experiment_metrics.csv)、[消融指標](results/final_presentation_completion/final_ablation_metrics.csv)、[配對 bootstrap](results/final_presentation_completion/final_bootstrap_comparisons.csv)、[CMS coverage](results/final_presentation_completion/cms_coverage_analysis.csv)、[CDC＋MEPS＋CMS 聯合結果](results/cms_meps_cdc_catboost_seed777/summary.csv)、[OOF 稽核](results/final_presentation_completion/source_audit.csv)。

## 研究協定與可重用資料

UCI 共 **99,343** 次就醫、**69,990** 位病人，seed **777**，病人互斥五折。所有納入的 OOF 均逐筆核對 Encounter ID、Patient ID、Fold、Label。AP 為 `average_precision_score`；表列六項指標為各測試折的**算術平均**。Precision、Recall、F1 固定取每折預測風險最高 10%。合併所有 OOF 計算的 pooled AP 存在 CSV 的另一欄，不與五折平均混淆。對照資料及 SHA256 見 `source_audit.csv`。

CCS 是 ICD-9 疾病分類表，單獨使用 CCS 不算加入患者級外部資料；正式的外部資料來源限定為 CDC BRFSS、MEPS、CMS HRRP。六項指標的方向為 PR-AUC (Average Precision, AP)、AUROC、Precision、Recall、F1 越高越好，Brier 越低越好。

先前論文報告的 0.237 AP 使用不同的單次 holdout；它只能作背景參考。公平比較以此處同一 99,343 筆、同五折 OOF 的 Rajpoot HGB 復現（AP 0.228703）及其他同切分模型為準。

已重用：論文復現 HGB、Raw HGB、CCS HGB、Raw CatBoost、MEPS、CDC、CMS 的既有五折 OOF；CMS 既有 2,000 次病人分折 bootstrap；MEPS 既有配對 bootstrap。新計算：Raw HGB＋Raw CatBoost 固定 50/50 融合、CMS covered 子群與缺少的配對 bootstrap。Ablation 只有重新訓練後才納入。

## 主要 Baseline 與 CCS 融合

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bhuvan_hgb | 0.228703 | 0.671776 | 0.272004 | 0.238729 | 0.254282 | 0.096042 |
| roy_hgb_46 | 0.237261 | 0.676968 | 0.279355 | 0.245182 | 0.261155 | 0.095577 |
| ccs_only_hgb | 0.239179 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 |
| catboost_raw | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 |
| raw_hgb_catboost_raw_50 | 0.246225 | 0.682744 | 0.287412 | 0.252253 | 0.268687 | 0.095119 |
| ccs_only_catboost_raw_50 | 0.247374 | 0.683446 | 0.288620 | 0.253313 | 0.269817 | 0.095068 |

### 新增 CCS CatBoost：兩個分類器都使用 CCS

原有 `ccs_only_catboost_raw_50` 只有 HGB 接收三個 CCS 診斷群組；[新實驗](results/ccs_catboost_5fold_seed777/summary.csv)讓 CatBoost 也接收 CCS。`catboost_raw_plus_ccs` 保留原始 `diag_1/2/3`，再加入 `ccs_diag_1/2/3`；`catboost_ccs_replace_diag` 以 CCS 取代三個原始診斷欄。HGB 一律使用原有 CCS-only HGB 的 OOF，兩模型以固定 50/50 機率融合。前者的完整 50/50 融合為 **0.247388**，比既有 0.247374 多 **0.000014**，病人分折配對 bootstrap 95% CI **[-0.000681, +0.000934]**，尚無穩定提升證據。[配對結果](results/ccs_catboost_5fold_seed777/paired_ap_bootstrap.csv)。

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `catboost_raw_plus_ccs` | 0.246917 | 0.682729 | 0.285096 | 0.250219 | 0.266521 | 0.095059 |
| `ccs_hgb_catboost_raw_plus_ccs_50` | **0.247388** | **0.683406** | **0.288016** | **0.252783** | **0.269252** | **0.095055** |
| `catboost_ccs_replace_diag` | 0.244229 | 0.679118 | 0.280866 | 0.246507 | 0.262567 | 0.095234 |
| `ccs_hgb_catboost_ccs_replace_diag_50` | 0.246786 | 0.681939 | 0.287714 | 0.252518 | 0.268969 | 0.095111 |

### LR、RF、XGBoost 的 CCS Replacement

[同切分重跑](results/classic_ccs_replacement_5fold_seed777/summary.csv)將三個原始診斷欄 `diag_1/2/3` 換成 `ccs_diag_1/2/3`，保留其餘 43 個欄位；使用相同 99,343 筆、seed 777 病人互斥五折。這是診斷表示法比較，**不算患者級 external data**。以下六項均是各測試折平均，Precision／Recall／F1 為每折 Top 10%；模型與門檻設定見 [run.json](results/classic_ccs_replacement_5fold_seed777/run.json)。

| CCS Replacement | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.213274 | 0.660160 | 0.260926 | 0.229007 | 0.243927 | 0.097085 |
| RF | 0.224593 | 0.672140 | 0.270292 | 0.237226 | 0.252682 | 0.097616 |
| XGBoost | 0.229885 | 0.674235 | 0.273817 | 0.240319 | 0.255977 | 0.095948 |

### 對兩個 CCS 模型分別使用 Class Weight／SMOTENC

[同切分實驗](results/ccs_ensemble_imbalance_5fold_seed777/summary.csv)保留每折既有模型參數，在各外層訓練資料內作加權或 SMOTENC（正負比 0.20），再於測試折平均機率；測試折沒有重採樣。下表六指標均以相同五折與 Top 10% 規則計算。Class weight 的 Brier 顯著大於未調整模型，兩種融合的 AP 均下降。[配對 bootstrap](results/ccs_ensemble_imbalance_5fold_seed777/paired_ap_bootstrap.csv)：Class Weight −基線 **−0.008274**，CI **[-0.010436, -0.006331]**；SMOTENC −基線 **−0.009884**，CI **[-0.012431, -0.007455]**。

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccs_hgb_catboost_raw_plus_ccs_50`（無調整） | 0.247388 | 0.683406 | 0.288016 | 0.252783 | 0.269252 | 0.095055 |
| HGB Class Weight | 0.231865 | 0.676006 | 0.276939 | 0.243061 | 0.258896 | 0.213740 |
| CatBoost Class Weight | 0.240472 | 0.683145 | 0.284290 | 0.249512 | 0.265768 | 0.215443 |
| HGB＋CatBoost Class Weight 50/50 | 0.239114 | 0.682805 | 0.283484 | 0.248806 | 0.265015 | 0.213780 |
| HGB SMOTENC | 0.224270 | 0.667673 | 0.270191 | 0.237138 | 0.252588 | 0.096965 |
| CatBoost SMOTENC | 0.239861 | 0.678902 | 0.285096 | 0.250221 | 0.266522 | 0.095455 |
| HGB＋CatBoost SMOTENC 50/50 | 0.237503 | 0.677175 | 0.282981 | 0.248364 | 0.264545 | 0.095756 |

### 既有 OOF 的融合比例探索

[權重實驗](results/hgb_catboost_blend_weights_baseline_seed777/summary.csv)只對上述**兩個都使用 CCS**的 HGB／CatBoost 已存 OOF 機率重算，未重新訓練。比例為 HGB／CatBoost。30/70 的 AP 最高，但比例是在同一批外層測試預測上比較，屬**探索性選擇**，不可視為獨立驗證後的最佳權重。所有比例的六項指標如下：

| HGB／CatBoost | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30/70 | **0.248238** | **0.683815** | 0.285599 | 0.250661 | 0.266992 | **0.095006** |
| 40/60 | 0.248036 | 0.683724 | 0.287210 | 0.252076 | 0.268499 | 0.095022 |
| 50/50 | 0.247388 | 0.683406 | 0.288016 | 0.252783 | 0.269252 | 0.095055 |
| 60/40 | 0.246502 | 0.682878 | **0.288419** | **0.253136** | **0.269628** | 0.095104 |
| 70/30 | 0.245193 | 0.682130 | 0.287915 | 0.252694 | 0.269157 | 0.095170 |

## 最終 External Data 單來源接法：CDC 1 欄、MEPS 2 欄、CMS 2 欄

此處的 **1／2／2** 指各來源**分別**加入 UCI CatBoost 的外部特徵數，不包含原始 46 欄、`ccs_diag_1/2/3` 三欄或 HGB 的輸入。三組單來源實驗均使用相同 seed 777 病人互斥五折、每折只用外層訓練資料重訓 UCI CatBoost；CCS HGB 使用同一份已保存 OOF。表中 Precision、Recall、F1 以各測試折風險最高 10% 計算。**目前沒有把這三個新版接法同時放進一個 CatBoost 重訓的 1／2／2 三來源結果**；後文的舊三來源聯合實驗使用不同的 CDC 與 MEPS 接法，列在歷史紀錄。

| 來源與實驗 | 外部資料如何對齊 UCI | 加到 CatBoost 的欄位 | 未匹配或缺值 | 來源實作／結果 |
| --- | --- | --- | --- | --- |
| **CDC BRFSS 2015：CatBoost Teacher（1 欄）** | 35,346 位有糖尿病的有效受訪者，以 `Age`, `Sex`, `HighBP`, `HighChol` 訓練 CatBoost 預測 `GenHlth ≥ 4`（自評健康尚可／差）。UCI 年齡十歲區間取中點、性別 female=0/male=1；`diag_1/2/3` 的 ICD-9 401–405 與 272 分別作高血壓／高血脂代理值。CDC 先以 seed 777 分層 80/20 驗證，再用全部合格 CDC 訓練一次；UCI 五折沿用同一 Teacher。 | `cdc_poor_health`：自評健康不佳的預測機率，**不是**再住院機率。原始 ICD-9 CatBoost 或原始 ICD-9＋CCS CatBoost 接收此欄；HGB 不接收。 | 無相應 ICD-9 證據的代理旗標為 0；不明年齡／性別可成缺值，由 CatBoost 處理。 | [CDC 接法](scripts/experiments/run_ccs_external_catboost_5fold.py)；[UCI 對齊](scripts/experiments/run_cdc_xgb.py)；[五折結果](results/ccs_cdc_catboost_teacher_5fold_seed777/summary.csv) |
| **MEPS 2008–2012 全疾病 Teacher（2 欄）** | 不限糖尿病，以住院事件日期推得同一人出院後第 1–30 天是否再住院。使用 13,178 筆標籤確定的事件（9,672 人，1,215 正例），連結 14,569 筆事件－CCS 紀錄；Teacher 的共同欄位為年齡、性別、種族、住院日數、至多 64 個 CCS 群組指示。UCI `diag_1/2/3` 任一落入群組即匹配。MEPS CatBoost Teacher 先以人員分組 80/20 驗證，再用全部 MEPS 訓練一次。 | `meps_30d_teacher_risk`：MEPS Teacher 對 UCI 的風險輸出；`meps_ccs_log_support = log1p(Σ UCI 匹配群組的 MEPS 事件數)`：群組樣本支持度，**不是**另一個風險率。兩欄加到原始 ICD-9＋CCS CatBoost；HGB 不接收。 | 未落入選取 CCS 群組時，所有群組指示為 0、support 為 0；Teacher 仍依其他共同欄位輸出風險。MEPS 原始缺值依共同欄位規則填補。 | [MEPS 對齊與公式](src/diabetes_project/meps_30d_transfer.py)；[五折實驗](scripts/experiments/run_meps_all_disease_30d_transfer.py)；[結果](results/meps_all_disease_30d_transfer_seed777/summary.csv) |
| **CMS HRRP（2 欄）** | 從四個全國風險標準化 30 天再住院措施 `READM_30_AMI/HF/PN/COPD` 取 `national_rate`，近似映到 CCS 100／108／122／127；**只以 UCI 主要診斷** `ccs_diag_1` 比對。命中 16,413／99,343 次就醫（16.5%）。 | `cms_readmission_rate = national_rate / 100`（疾病別全國彙總率，非個人機率）；`cms_mapped`（是否命中）。原始 ICD-9 CatBoost 或原始 ICD-9＋CCS CatBoost 接收兩欄；HGB 不接收。 | 未命中時分別填 `-1`、`0`。來源醫院列的 `score` 不用於此特徵。 | [CMS 準備](scripts/experiments/prepare_cms_readmission_prior.py)；[五折接法](scripts/experiments/run_ccs_external_catboost_5fold.py)；[結果](results/ccs_external_catboost_5fold_seed777/summary.csv) |

最終單來源的固定融合均為 `0.5 × ccs_only_hgb + 0.5 × 該來源增強的 CatBoost`，直接平均 OOF 預測機率，沒有在融合前另作校準。CDC／CMS 有「原始 46 欄 CatBoost」和「原始 46 欄＋三個 CCS 欄位 CatBoost」兩種基底；MEPS 全疾病 Teacher 使用後者。**配對比較只對同一基底**：原始 CatBoost 融合對 0.247374 基線；兩模型都用 CCS 的融合對 0.247388 基線。下表列出各自的完整六項五折平均指標。

| 單來源融合模型 | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 無外部資料：CCS HGB＋原始 CatBoost 50/50 | 0.247374 | 0.683446 | 0.288620 | 0.253313 | 0.269817 | 0.095068 |
| CDC CatBoost Teacher 1 欄＋原始 CatBoost 50/50 | 0.246785 | 0.683479 | 0.289124 | 0.253755 | 0.270287 | 0.095089 |
| CMS 2 欄＋原始 CatBoost 50/50 | 0.247074 | 0.683315 | 0.289225 | 0.253844 | 0.270382 | 0.095072 |
| 無外部資料：CCS HGB＋CCS CatBoost 50/50 | 0.247388 | 0.683406 | 0.288016 | 0.252783 | 0.269252 | 0.095055 |
| CDC CatBoost Teacher 1 欄＋CCS CatBoost 50/50 | **0.247640** | 0.683455 | 0.288520 | 0.253225 | 0.269722 | 0.095047 |
| CMS 2 欄＋CCS CatBoost 50/50 | 0.247473 | 0.683327 | 0.288218 | 0.252959 | 0.269440 | 0.095047 |
| 全疾病 MEPS Teacher 2 欄＋CCS CatBoost 50/50 | 0.247526 | 0.683539 | 0.288721 | 0.253402 | 0.269911 | **0.095043** |

相對**各自同架構基線**的平均 ΔAP／病人分折配對 bootstrap 95% CI：CDC＋CCS CatBoost **+0.000252**／**[-0.000372, +0.000807]**；CMS＋CCS CatBoost **+0.000085**／**[-0.000486, +0.000627]**；全疾病 MEPS Teacher＋CCS CatBoost **+0.000138**／**[-0.000494, +0.000712]**。三個 CI 都跨零，不能宣稱已證實整體 AP 改善。[CDC 配對](results/ccs_cdc_catboost_teacher_5fold_seed777/paired_ap_bootstrap.csv)｜[CMS 配對](results/ccs_external_catboost_5fold_seed777/paired_ap_bootstrap.csv)｜[MEPS 配對](results/meps_all_disease_30d_transfer_seed777/paired_ap_bootstrap.csv)。

### MEPS 2008–2012：全疾病 30 天 Teacher 與 Joint Training

[全疾病實驗](results/meps_all_disease_30d_transfer_seed777/summary.csv)使用 13,178 筆標籤可確定的事件、9,672 人、1,215 個 30 天再住院正例。由同一人出院後第 1–30 天是否再次住院建標籤，這是事件日期推得的標籤。已連結 14,569 筆不重複事件－CCS 紀錄；從中選最多 64 個群組。原本糖尿病限定的 2,688 筆事件在全疾病表中均能對齊且標籤一致。[來源與設定](results/meps_all_disease_30d_transfer_seed777/config.json)。

Teacher 路徑用 MEPS 年齡、性別、種族、住院日數與 CCS 群組訓練 CatBoost；對 UCI 每筆推得 `meps_30d_teacher_risk`，另算 `meps_ccs_log_support = log1p(符合 UCI 診斷 CCS 的 MEPS 事件數總和)`。這兩欄加入**原始 ICD-9＋CCS CatBoost**，再與不變的 CCS HGB 以 50/50 融合。Joint Training 路徑則以 UCI 和 MEPS 的共同欄位訓練共享網路，各自使用自己的標籤與輸出頭；沒有固定的 Teacher 欄位。糖尿病 teacher 對照也在相同 UCI 折及 CatBoost 輸入架構下重訓。這個無外部資料基線是 **0.247388**，與上面的原始 46 欄 CatBoost 基線 0.247374 不可互換。[接法與程式行號](results/final_external_data_audit/EXTERNAL_DATA_FINAL_AUDIT.md)。

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccs_hgb_catboost_raw_plus_ccs_50`（無 MEPS） | 0.247388 | 0.683406 | 0.288016 | 0.252783 | 0.269252 | 0.095055 |
| `diabetes_meps_teacher_catboost` | 0.246924 | 0.682517 | 0.285498 | 0.250573 | 0.266898 | 0.095052 |
| `diabetes_meps_teacher_hgb50` | 0.247484 | 0.683300 | 0.287210 | 0.252075 | 0.268498 | 0.095052 |
| `all_disease_meps_teacher_catboost` | 0.247298 | 0.682883 | 0.286002 | 0.251015 | 0.267369 | 0.095035 |
| `all_disease_meps_teacher_hgb50` | **0.247526** | **0.683539** | **0.288721** | **0.253402** | **0.269911** | **0.095043** |
| `uci_only_nn` | 0.218535 | 0.665779 | 0.263646 | 0.231393 | 0.246468 | 0.097122 |
| `all_disease_joint_meps_30d` | 0.219110 | 0.666897 | 0.266062 | 0.233515 | 0.248729 | 0.097045 |
| `uci_only_nn_catboost50` | 0.239306 | 0.681732 | 0.282377 | 0.247833 | 0.263979 | 0.095437 |
| `all_disease_joint_meps_30d_catboost50` | 0.239413 | 0.681475 | 0.286002 | 0.251016 | 0.267369 | 0.095388 |

[病人分折配對 bootstrap](results/meps_all_disease_30d_transfer_seed777/paired_ap_bootstrap.csv)：全疾病 Teacher 50/50 −無 MEPS 基線 **+0.000138**，95% CI **[-0.000494, +0.000712]**；全疾病 −糖尿病限定 Teacher 50/50 **+0.000041**，CI **[-0.000663, +0.000704]**；全疾病 Joint NN −UCI-only NN **+0.000574**，CI **[-0.002020, +0.003216]**。三者均跨零，未確立改善。下方歷史區段的糖尿病限定 MEPS 統計特徵屬另一種接法，不可混用結果。

## CatBoost 原始特徵消融

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier | Full_minus_Ablated_PR_AUC |
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

CMS 只依 UCI 主要診斷 CCS 對應四個概念：心肌梗塞、心衰竭、肺炎、COPD。命中率為 **16,413/99,343（16.5%）**。數值是 CMS 全國風險標準化再住院率，並非個人標籤。早期 `ccs_only_hgb_cms_catboost_50` 是 CCS HGB 與 **原始 46 欄＋CMS 兩欄的 CatBoost** 以 50/50 平均，AP **0.247074**；新版讓 CatBoost 同時接收 CCS 與 CMS 兩欄，`ccs_hgb_ccs_catboost_cms_50` 的 AP 為 **0.247473**。兩者基底不同，應各自與對應的無 CMS 基線配對。

子群分析是在既有五折上探索；各子群的 Precision／Recall／F1 重新以該子群內最高 10% 評估，不能與全體門檻數字直接替換。正例率及樣本數列於下表：

| Subset | Model | N | Positive_Count | Positive_Rate | PR-AUC (Average Precision, AP) |
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

## External Data 最終選擇：採用全疾病 MEPS Teacher

最終簡報採用 **MEPS 2008–2012 全疾病 30 天 Teacher** 作為外部資料接法：將 `meps_30d_teacher_risk`、`meps_ccs_log_support` 兩欄加入保留原始 ICD-9 與 CCS 的 CatBoost，再與同折 CCS HGB 預測機率固定 50/50 融合。這是上方「1／2／2」各來源單獨測試後選定的 **MEPS 兩欄版本**；CDC CatBoost Teacher 與 CMS 兩欄保留為對照，不把舊三來源聯合實驗當作此版本的結果。

此模型五折平均 PR-AUC (Average Precision, AP) **0.247526**、AUROC **0.683539**、Precision **0.288721**、Recall **0.253402**、F1 **0.269911**、Brier **0.095043**。相同 HGB＋CatBoost 架構、未加 MEPS 的基線 AP 為 **0.247388**；平均差 **+0.000138**，病人分折配對 bootstrap 95% CI **[-0.000494, +0.000712]**。因此 MEPS 是**正式呈現採用的外部資料**，但尚不能宣稱它顯著提升整體 AP。[結果](results/meps_all_disease_30d_transfer_seed777/summary.csv)｜[配對 bootstrap](results/meps_all_disease_30d_transfer_seed777/paired_ap_bootstrap.csv)。

CDC CatBoost Teacher 的同架構融合 AP **0.247640**，點估計略高於 MEPS 的 **0.247526**；兩者來自不同外部訊號，且各自對基線的 CI 都跨零，不能把這個微小差距解讀為可靠排序。選擇 MEPS 的理由是其外部標籤直接由 **30 天再住院事件**推得，與研究目標較接近；這是方法選擇，並非已證實的效能優勢。MEPS 與 UCI 的族群、事件日期可觀測性和標籤邊界仍有差異，這些可能限制轉移效果，但尚未完成因果驗證。

## 可以直接放入 Final Presentation 的文字

> 在相同的 99,343 次就醫、69,990 位病人、seed 777 病人互斥五折下，Raw ICD-9 CatBoost 的平均 PR-AUC (Average Precision, AP) 為 0.246506；兩個分類器均使用 CCS 的固定 50/50 融合為 **0.247388**。最終外部資料選用 **MEPS 2008–2012 全疾病 30 天 Teacher**：它提供再住院風險分數與 CCS 群組支持度兩欄，加入 CatBoost 後再與 CCS HGB 50/50 融合，AP 為 **0.247526**。相對同架構無 MEPS 基線的 ΔAP 為 **+0.000138**，病人配對 95% CI **[-0.000494, +0.000712]**；因此目前只能說 MEPS 已實際接入，不能宣稱整體預測效益已獲統計支持。CDC CatBoost Teacher 一欄與 CMS 兩欄均另作單來源對照；先前使用不同接法的三來源聯合結果放在歷史紀錄。

## 尚缺或不適用

無。

## 重現命令

```bash
cd /media/synology/G/314554044/diabetes_project
python scripts/experiments/run_final_catboost_ablation.py --ablations E1_no_number_inpatient E2_no_raw_diagnoses E3_no_all_diagnosis_derived --threads 8
python scripts/experiments/complete_final_presentation.py --bootstrap 2000
```

本次補做結果的入口如下；除權重與校準分析讀取已存 OOF，其餘為訓練實驗，需先備妥本地原始資料與既有 OOF。完整參數、來源路徑與 seed 另存於各結果目錄的 `run.json` 或 `config.json`。

```bash
python scripts/experiments/run_ccs_catboost_5fold.py --max-folds 5 --seed 777
python scripts/experiments/run_ccs_ensemble_imbalance_5fold.py --max-folds 5 --seed 777
python scripts/experiments/run_meps_all_disease_30d_transfer.py --include-joint --max-folds 5 --seed 777
python scripts/experiments/analyze_hgb_catboost_blend_weights.py --outdir results/hgb_catboost_blend_weights_baseline_seed777 --bootstrap 0 --seed 777
python scripts/experiments/analyze_calibration_5fold.py --bins 10 --bootstrap 2000 --seed 777
```

模型設定：CatBoost CPU、原有 46 個出院時特徵、`depth` 4／6、`iterations` 250／500／750、learning rate 0.035、L2 5／15；內層每個 outer train 用 20% 病人互斥驗證，按 AP 選參，tie-break Brier；最終全 outer train 重訓。詳細設定見 [CatBoost 消融設定](results/final_presentation_completion/catboost_ablation_run.json)，原始模型設定見 [論文比較設定](results/paper_fair_comparison_seed777/run.json)。

# 延伸實驗與歷史紀錄

以下保留先前已完成的其他五折方法，包括 NHANES、不同 seed 與 99,340 筆舊隊列。這些結果供方法歷程查閱；不同隊列、切分或門檻策略不得與前述正式主表直接排名。重複的核心模型以最前方的最新 OOF 重算表為準。

## 歷史 External Data 接法：舊 CDC、MEPS Statistical 與三來源聯合

以下保留已完成的舊版五折結果；CDC 使用 XGBoost Teacher，MEPS 使用四個 CCS 群組統計欄位。它們不是上方 CDC CatBoost Teacher／全疾病 MEPS Teacher 的 1／2／2 接法，也沒有測得新版三來源聯合結果。

### CMS HRRP

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cms_hgb | 0.238496 | 0.678059 | 0.281873 | 0.247390 | 0.263508 | 0.095499 |
| cms_catboost_raw | 0.246507 | 0.682518 | 0.286103 | 0.251103 | 0.267463 | 0.095083 |
| cms_hgb_catboost_raw_50 | 0.246741 | 0.683289 | 0.287714 | 0.252517 | 0.268969 | 0.095083 |
| ccs_only_hgb_cms_catboost_50 | 0.247074 | 0.683315 | 0.289225 | 0.253844 | 0.270382 | 0.095072 |
| cms_hgb_cms_catboost_50 | 0.246644 | 0.683144 | 0.290232 | 0.254727 | 0.271323 | 0.095088 |

### CDC BRFSS

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2_Clinical | 0.234919 | 0.678280 | 0.276334 | 0.242530 | 0.258331 | 0.095664 |
| E4_Clinical_CDCHealth | 0.235256 | 0.678403 | 0.276939 | 0.243060 | 0.258896 | 0.095640 |
| Only_DiffWalk | 0.235084 | 0.678039 | 0.277039 | 0.243148 | 0.258990 | 0.095663 |
| No_DiffWalk | 0.234110 | 0.677909 | 0.274824 | 0.241204 | 0.256919 | 0.095693 |
| Permuted_DiffWalk | 0.235028 | 0.678541 | 0.274622 | 0.241026 | 0.256730 | 0.095662 |

### MEPS

| Model | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ccs_meps_group_hgb | 0.239077 | 0.677884 | 0.282276 | 0.247745 | 0.263885 | 0.095500 |
| ccs_meps_catboost_raw_50 | 0.247139 | 0.683233 | 0.286103 | 0.251104 | 0.267463 | 0.095087 |
| uci_only_nn | 0.216620 | 0.664481 | 0.263243 | 0.231040 | 0.246093 | 0.097245 |
| joint_meps_30d | 0.219264 | 0.665757 | 0.263746 | 0.231482 | 0.246563 | 0.097049 |
| joint_meps_30d_catboost50 | 0.239537 | 0.680805 | 0.283686 | 0.248983 | 0.265204 | 0.095406 |

### CDC＋MEPS＋CMS 三來源聯合實驗

這組已完成 **99,343 筆、69,990 位病人、seed 777、相同病人互斥五折**。所有模型的 Precision、Recall、F1 都固定取每折風險最高 10%；其餘三項使用原始預測機率。單一及雙來源列使用同一 Raw ICD-9 CatBoost 架構做配對控制；`cms_meps_cdc_hgb50` 再與 CCS-only HGB 的機率固定 50/50 融合。下表中的 `ccs_only_hgb_catboost_raw_50` 與前文 `ccs_only_catboost_raw_50` 是同一基線預測。[完整六項指標](results/cms_meps_cdc_catboost_seed777/summary.csv)｜[逐折指標](results/cms_meps_cdc_catboost_seed777/fold_metrics.csv)｜[實驗稽核](results/cms_meps_cdc_catboost_seed777/audit.json)。

| 模型 | PR-AUC (Average Precision, AP) | AUROC | Precision | Recall | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `catboost_raw` | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 |
| `ccs_only_hgb_catboost_raw_50` | 0.247374 | 0.683446 | 0.288620 | 0.253313 | 0.269817 | 0.095068 |
| `cms_catboost_raw` | 0.246507 | 0.682518 | 0.286103 | 0.251103 | 0.267463 | 0.095083 |
| `meps` | 0.246960 | 0.683091 | 0.287815 | 0.252605 | 0.269063 | 0.095054 |
| `cdc` | 0.246266 | 0.682434 | 0.284693 | 0.249865 | 0.266144 | 0.095104 |
| `cms_meps` | 0.246276 | 0.682878 | 0.286808 | 0.251722 | 0.268122 | 0.095085 |
| `cms_cdc` | 0.246211 | 0.682839 | 0.284592 | 0.249777 | 0.266051 | 0.095086 |
| `meps_cdc` | 0.247036 | 0.683011 | 0.288218 | 0.252959 | 0.269440 | 0.095057 |
| `cms_meps_cdc` | 0.246367 | 0.683025 | 0.287613 | 0.252429 | 0.268875 | 0.095080 |
| `cms_meps_cdc_hgb50` | 0.247091 | 0.683464 | 0.291138 | 0.255522 | 0.272170 | 0.095073 |

三來源融合相對無外部資料的 CCS HGB＋Raw CatBoost，平均 **ΔAP -0.000283**；每折內按病人配對抽樣 2,000 次的 95% CI 為 **[-0.000993, +0.000543]**。Top 10% 的 F1 從 **0.269817** 到 **0.272170**，但 AP 較低且配對 CI 跨零，不能宣稱三來源聯合提高主要 PR-AUC (Average Precision, AP)。其他配對差值見[完整 bootstrap](results/cms_meps_cdc_catboost_seed777/paired_ap_bootstrap.csv)。

CDC 使用 BRFSS 糖尿病受訪者的**自評健康不佳代理分數**，它本身不是再住院標籤；MEPS 使用精確日期事件的 CCS 群組 30 天再就醫／再住院統計；CMS 使用四類疾病的全國彙總再住院率，約覆蓋醫院隊列 16.5%。這些欄位以共同人口或診斷資訊對齊；不同資料來源的測量方式與涵蓋範圍，可能限制外部訊號的效益，但本實驗不能判定因果原因。

#### 三來源 CatBoost 的實際接法

以下對應[實作稽核與程式行號](results/final_external_data_audit/EXTERNAL_DATA_FINAL_AUDIT.md)，不是不同時期 CDC encoder 或 MEPS joint 模型的接法。三來源腳本的 CatBoost 保留原始 **46 個出院時欄位**，按組合附加外部特徵；固定 CCS-only HGB 保留 46 欄及三個 CCS 診斷類別。各外部特徵使用來源資料的標籤或公開彙總，不用 UCI 測試折標籤建構；UCI 分類器每折只用外層訓練資料擬合。

| 來源 | 外部輸入與對齊 | 真正附加至 UCI CatBoost 的欄位 | 未匹配時 |
| --- | --- | --- | --- |
| CDC BRFSS | 2015 年糖尿病受訪者的 `Age`, `Sex`, `HighBP`, `HighChol` 預測 `GenHlth≥4`；UCI 用年齡區間中點、性別，並從 `diag_1/2/3` 的 ICD-9 401–405、272 建血壓／膽固醇代理值。 | `cdc_poor_health`：CDC XGBoost 的自評健康不佳機率；這條主實驗沒有 DiffWalk 或 embedding 欄位。 | 個別 bridge 欄缺失由 XGBoost 處理；無病碼證據的代理旗標為 0。 |
| MEPS Statistical | 2008–2012 糖尿病事件按**同一事件**連結 CCS；每個至少 30 筆的群組以 `(群組正例數＋50×全事件正例率)/(群組事件數＋50)` 得 30 天再住院與急診再訪率；UCI 的三個診斷先轉 AHRQ CCS。 | `meps_ccs_readmit_mean`, `meps_ccs_revisit_mean`, `meps_ccs_readmit_max`, `meps_ccs_revisit_max`：三個診斷位置的群組率均值與最大值。 | 無對應 CCS 或不足 30 筆群組，用 MEPS 全事件平均率。 |
| CMS HRRP | 四個全國措施 `READM_30_AMI/HF/PN/COPD` 各映射一個 CCS 群組，只用 UCI **主要診斷** `ccs_diag_1` 匹配；來源值是國家疾病別率，並非個人風險。 | `cms_readmission_rate= national_rate/100`, `cms_mapped`。住院機構層級 `score` 沒有用於模型或加權。 | 兩欄分別為 `-1`, `0`。 |

單一、雙來源的 `meps`／`cdc`／`cms_meps`／`cms_cdc`／`meps_cdc` 和三來源 `cms_meps_cdc` 都是**只改 CatBoost 輸入**，沒有 HGB。`cms_meps_cdc_hgb50 = 0.5 × ccs_only_hgb + 0.5 × cms_meps_cdc`，直接平均既有預測機率，融合前沒有另做機率校準。歷史 `ccs_meps_catboost_raw_50` 則把 MEPS 四個統計欄位**加到 HGB**，與沒有 MEPS 欄位的 Raw CatBoost 平均；不能與本段 `meps` 當成同一實驗。

聯合 CatBoost 的五折 SHAP 每折抽 1,000 筆，共 5,000 筆。原始醫院特徵 `number_inpatient`、`discharge_disposition_id`、`diag_1` 仍排前三；外部特徵如下。[完整 SHAP 排名](results/cms_meps_cdc_catboost_seed777/shap_global.csv)｜[各折 SHAP](results/cms_meps_cdc_catboost_seed777/shap_fold.csv)。SHAP 表示模型使用了特徵，不能單靠它證明外部資料增加預測效益。

| 外部特徵 | 平均絕對 SHAP（log-odds） | 排名 |
| --- | ---: | ---: |
| `meps_ccs_revisit_mean` | 0.019719 | 15 |
| `meps_ccs_readmit_max` | 0.012726 | 20 |
| `meps_ccs_revisit_max` | 0.010789 | 22 |
| `meps_ccs_readmit_mean` | 0.010704 | 23 |
| `cdc_poor_health` | 0.007321 | 24 |
| `cms_readmission_rate` | 0.003490 | 35 |
| `cms_mapped` | 0.000835 | 44 |

同一組 OOF 的固定警示比例敏感度分析如下；**PR-AUC (Average Precision, AP)、AUROC、Brier 不隨警示比例改變**。三來源融合在 Top 5% 與 10% 的 F1 略高，Top 15%～25% 則沒有此優勢。[全部模型與比例的六項指標](results/top_fractions_cms_meps_cdc_5fold_seed777/summary.csv)。

| 每折警示比例 | 無外部資料融合 F1 | 三來源融合 F1 |
| --- | ---: | ---: |
| Top 5% | 0.216475 | 0.218440 |
| Top 10% | 0.269817 | 0.272170 |
| Top 15% | 0.289692 | 0.288166 |
| Top 20% | 0.293723 | 0.293466 |
| Top 25% | 0.293950 | 0.293728 |

#### 五折機率可靠度：Calibration、ECE、Brier

[校準分析](results/calibration_analysis_5fold_seed777/CALIBRATION_ANALYSIS_SUMMARY.md)直接使用已存 OOF 原始機率，沒有重新訓練或重校準；每折十個等頻箱，定量 ECE 比較在同折使用一致箱界。這是**機率品質**分析，與上表的 Top 10% 門檻分類不同。ECE 依分箱方式而變；Brier 同時反映辨別能力，單靠它不能證明機率已充分校準。[逐折與 pooled 數值](results/calibration_analysis_5fold_seed777/calibration_metrics.csv)｜[分箱統計](results/calibration_analysis_5fold_seed777/calibration_bin_statistics.csv)。

| Model | 五折平均 Brier | 五折平均 ECE | Pooled OOF ECE |
| --- | ---: | ---: | ---: |
| Raw ICD-9 CatBoost | 0.095084 | 0.006964 | 0.004801 |
| CCS HGB＋Raw CatBoost 50/50 | **0.095068** | 0.007441 | 0.004851 |
| CDC＋MEPS＋CMS CatBoost＋CCS HGB 50/50 | 0.095073 | 0.007280 | 0.005360 |

外部融合 −原融合的病人分折配對 bootstrap：ΔBrier **+0.000006**，95% CI **[-0.000036, +0.000053]**；ΔECE **−0.000161**，CI **[-0.001567, +0.001684]**。兩者均跨零，無法確認外部融合改善機率品質。[配對數值](results/calibration_analysis_5fold_seed777/calibration_paired_bootstrap.csv)；[pooled](results/calibration_analysis_5fold_seed777/calibration_curve_pooled.png)與[五折](results/calibration_analysis_5fold_seed777/calibration_curve_fivefold.png)圖均用原始機率。

### 早期 External Data 比較與限制分析


下表保留先前的同架構配對比較；兩個模型各自的六項指標見本歷史區段與[原始完整配對表](results/final_presentation_completion/external_data_comparisons.csv)。此表不代表上方採用的全疾病 MEPS Teacher 接法。

| 來源 | 原始模型 | 外部模型 | ΔAP | 95% CI |
| --- | --- | --- | ---: | ---: |
| CDC BRFSS | `E2_Clinical` | `E4_Clinical_CDCHealth` | +0.000337 | [-0.000653, +0.001471] |
| CDC BRFSS | `Permuted_DiffWalk` | `Only_DiffWalk` | +0.000056 | [-0.000894, +0.001066] |
| MEPS | `ccs_only_catboost_raw_50` | `ccs_meps_catboost_raw_50` | -0.000234 | [-0.001096, +0.000547] |
| MEPS | `uci_only_nn` | `joint_meps_30d` | +0.002644 | [+0.000035, +0.005273] |
| CMS HRRP | `ccs_only_hgb` | `cms_hgb` | -0.000683 | [-0.001756, +0.000367] |
| CMS HRRP | `catboost_raw` | `cms_catboost_raw` | +0.000001 | [-0.001165, +0.001238] |
| CMS HRRP | `ccs_only_catboost_raw_50` | `cms_hgb_catboost_raw_50` | -0.000633 | [-0.001170, -0.000063] |
| CMS HRRP | `ccs_only_catboost_raw_50` | `ccs_only_hgb_cms_catboost_50` | -0.000300 | [-0.000908, +0.000487] |

#### 資料來源、對齊方式與限制

| 來源 | 使用的資訊與對齊方式 | 主要限制 |
| --- | --- | --- |
| CDC BRFSS | 健康問卷中的慢性病、自評健康及 DiffWalk 訊號，經共通人口／共病欄位對齊 UCI 臨床特徵；同架構的 Clinical 與 Clinical＋CDC 比較。 | 問卷測量與住院 EHR 不同；部分特徵只能映射到疾病群組，缺乏患者實測值。 |
| MEPS | 年度利用率、30 天事件及其 ICD-9／CCS 疾病群組統計，按診斷群組映射到 UCI；共同訓練另以 UCI-only NN 配對。 | 年度標籤與 30 天再住院不等價；事件群組樣本較少，與 UCI 既有利用史可能重疊。 |
| CMS HRRP | 四種疾病的全國彙總 30 天再住院率，以 UCI 主要診斷 CCS 映射；未命中者保留缺失／對應旗標。 | 只覆蓋 16.5%，沒有個人級資料，疾病類別有限；本次 CMS 指標期間為 2023–2025，與歷史 UCI 住院期間不同。 |

- **CDC：已觀察** Clinical 基線 AP **0.234919**，加入 CDC 健康訊號 **0.235256**（差 **+0.000337**），DiffWalk 真值與置換對照接近。**可能原因**是問卷的自評健康與 UCI 住院紀錄測量方式不同，且映射的共病旗標不等於真實病史。這些是 domain shift 與個人化不足的可能解釋，**不是已證明的因果原因**。CDC 沒有再住院標籤。
- **MEPS：已觀察** CCS HGB＋Raw CatBoost AP **0.247374**，換為 MEPS HGB＋Raw CatBoost **0.247139**；共同訓練 NN 從 **0.216620** 到 **0.219264**（差 **+0.002644**，原配對 CI 見表）。**可能原因**是年度利用率與 30 天標籤不完全相同，以及外部利用訊號與 `number_inpatient` 重疊；這些尚未被正式因果分解。共同訓練的小幅正差支持 MEPS 在該神經網路設定下帶來可用訊號，但未使目前最佳組合改善。
- **CMS：當時已觀察** 四類彙總率覆蓋 16.5%；當時採原始 CatBoost 的 CMS 融合 AP **0.247074**，低於對應基準 **0.247374**。CMS HGB＋原 CatBoost 的配對 CI 完全小於零。新版 CCS CatBoost＋CMS 融合另見上方單來源表。**可能原因**是來源只有四類群組、沒有個人狀態，且疾病訊息與原有 ICD-9／CCS 重疊。是否在某些年齡或疾病子群較有用仍屬未驗證假設；本次子群表僅供探索。

## 先前論文復現、標準基線與外部資料延伸

下表均屬近期相同五折，六項指標使用同一個最高 10% 警示規則；重複的 OOF 機率只列一次。論文方法的 `bhuvan_hgb` 是在**本專題五折**重跑的復現對照，不是搬用論文單次測試數字。

### 論文復現、CatBoost 與 MEPS 融合

| Experiment | Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_only_catboost_raw_50 | 0.247374 | 0.683446 | 0.288620 | 0.253313 | 0.269817 | 0.095068 | 0.245932 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_only_catboost_50 | 0.247337 | 0.681428 | 0.287815 | 0.252606 | 0.269063 | 0.095093 | 0.246005 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_meps_catboost_50 | 0.247140 | 0.681210 | 0.288318 | 0.253048 | 0.269534 | 0.095118 | 0.245854 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_meps_catboost_raw_50 | 0.247139 | 0.683233 | 0.286103 | 0.251104 | 0.267463 | 0.095087 | 0.245889 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | catboost_raw | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 | 0.245430 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | catboost_family | 0.244454 | 0.677482 | 0.284089 | 0.249336 | 0.265580 | 0.095233 | 0.243281 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | bhuvan_hgb_catboost_50 | 0.241891 | 0.677469 | 0.283082 | 0.248451 | 0.264638 | 0.095395 | 0.240679 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_only_hgb | 0.239179 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 | 0.237823 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | ccs_meps_group_hgb | 0.239077 | 0.677884 | 0.282276 | 0.247745 | 0.263885 | 0.095500 | 0.238043 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | roy_hgb_46 | 0.237261 | 0.676968 | 0.279355 | 0.245182 | 0.261155 | 0.095577 | 0.235940 |
| [paper_fair_comparison_seed777](results/paper_fair_comparison_seed777/summary.csv) | bhuvan_hgb | 0.228703 | 0.671776 | 0.272004 | 0.238729 | 0.254282 | 0.096042 | 0.227821 |


同一折病人配對 bootstrap 顯示，`catboost_raw` 相對 `ccs_only_hgb` 的平均 AP 差為 **+0.007327**（95% CI **[+0.004039, +0.010484]**）。但 `ccs_meps_catboost_50` 相對同架構 `ccs_only_catboost_50` 為 **−0.000197**（95% CI **[−0.001026, +0.000604]**），加入 MEPS 的額外增益未獲證實。[配對分析](results/paper_fair_comparison_seed777/paired_ap_bootstrap.csv)。

### LR、RF、XGBoost 與類別不平衡

| Experiment | Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_standard | 0.230520 | 0.673929 | 0.273817 | 0.240320 | 0.255977 | 0.095958 | 0.229276 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | rf_standard | 0.222174 | 0.670451 | 0.267976 | 0.235194 | 0.250517 | 0.098313 | 0.221440 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_weighted | 0.221244 | 0.668693 | 0.263041 | 0.230865 | 0.245905 | 0.224077 | 0.218146 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | lr_standard | 0.212744 | 0.656468 | 0.261128 | 0.229185 | 0.244116 | 0.097284 | 0.212099 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | lr_balanced | 0.210895 | 0.654575 | 0.258107 | 0.226534 | 0.241292 | 0.220178 | 0.210451 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_smotenc_020 | 0.206220 | 0.645952 | 0.244310 | 0.214424 | 0.228394 | 0.097684 | 0.205295 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | rf_balanced | 0.202444 | 0.664644 | 0.242900 | 0.213186 | 0.227075 | 0.226550 | 0.201813 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_smotenc_035 | 0.196266 | 0.630477 | 0.241793 | 0.212213 | 0.226039 | 0.099883 | 0.195528 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_smotenc_050 | 0.186172 | 0.614392 | 0.238066 | 0.208944 | 0.222556 | 0.104912 | 0.183114 |
| [classic_5fold_recheck_seed777](results/classic_5fold_recheck_seed777/summary.csv) | xgb_smotenc_100 | 0.171574 | 0.592494 | 0.237563 | 0.208501 | 0.222085 | 0.137953 | 0.171553 |

標準 XGBoost 對標準 RF、標準 LR 的平均 AP 差依序為 +0.008345、+0.017776；加權 XGBoost 相對標準 XGBoost 為 −0.009276。SMOTENC 的 0.20、0.35、0.50、1.00 比例越高，AP 依序約 0.206、0.196、0.186、0.172。加權 LR／RF／XGBoost 的 Brier 約 0.22，直接輸出的機率沒有良好校準。

### MEPS＋NHANES 雙來源與 30 天共同訓練

| Experiment | Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | group_teacher_catboost_hgb50 | 0.247038 | 0.683300 | 0.288218 | 0.252959 | 0.269440 | 0.095077 | 0.245591 |
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | group_teacher_catboost | 0.246291 | 0.682596 | 0.285498 | 0.250572 | 0.266898 | 0.095091 | 0.245139 |
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | joint_meps_30d_catboost50 | 0.239537 | 0.680805 | 0.283686 | 0.248983 | 0.265204 | 0.095406 | 0.238232 |
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | uci_only_nn_catboost50 | 0.238045 | 0.680662 | 0.280262 | 0.245978 | 0.262003 | 0.095510 | 0.236593 |
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | joint_meps_30d | 0.219264 | 0.665757 | 0.263746 | 0.231482 | 0.246563 | 0.097049 | 0.217643 |
| [meps_30d_joint_transfer_seed777](results/meps_30d_joint_transfer_seed777/summary.csv) | uci_only_nn | 0.216620 | 0.664481 | 0.263243 | 0.231040 | 0.246093 | 0.097245 | 0.214873 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_dual_catboost_raw50 | 0.246877 | 0.683299 | 0.287613 | 0.252429 | 0.268875 | 0.095096 | 0.245526 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_meps_catboost_raw50 | 0.246751 | 0.683111 | 0.288922 | 0.253578 | 0.270099 | 0.095096 | 0.245309 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_nhanes_catboost_raw50 | 0.246684 | 0.683311 | 0.288218 | 0.252960 | 0.269440 | 0.095099 | 0.245349 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_nhanes_catboost_raw50 | 0.246682 | 0.683301 | 0.288419 | 0.253136 | 0.269628 | 0.095095 | 0.245270 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_dual_catboost_raw50 | 0.246639 | 0.683150 | 0.287311 | 0.252164 | 0.268593 | 0.095102 | 0.245246 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_meps_catboost_raw50 | 0.246614 | 0.683273 | 0.287513 | 0.252341 | 0.268781 | 0.095096 | 0.245265 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_dual | 0.238303 | 0.677964 | 0.281269 | 0.246861 | 0.262944 | 0.095536 | 0.236917 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_nhanes | 0.237939 | 0.678102 | 0.284592 | 0.249778 | 0.266051 | 0.095526 | 0.236756 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_nhanes | 0.237839 | 0.677914 | 0.281772 | 0.247303 | 0.263415 | 0.095541 | 0.236538 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_dual | 0.237757 | 0.677639 | 0.279859 | 0.245623 | 0.261626 | 0.095551 | 0.236425 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | multitask_meps | 0.237703 | 0.677568 | 0.281974 | 0.247480 | 0.263603 | 0.095547 | 0.236463 |
| [meps_nhanes_dual_external_seed777](results/meps_nhanes_dual_external_seed777/summary.csv) | grouping_meps | 0.237652 | 0.678020 | 0.280765 | 0.246418 | 0.262473 | 0.095536 | 0.236549 |

外部 30 天共同訓練 `joint_meps_30d` 相對 `uci_only_nn` 的平均 AP 差 **+0.002644**，病人配對 bootstrap 95% CI **[+0.000035, +0.005273]**；加上 CatBoost 後仍低於最佳 CCS-only 融合。雙來源最佳融合的增益相對最佳 CCS-only 融合未獲證實。[配對檔](results/meps_30d_joint_transfer_seed777/paired_ap_bootstrap.csv)、[雙來源結果](results/meps_nhanes_dual_external_seed777/summary.csv)。

## 近期同切分其餘完整五折實驗

包含 CDC、MEPS 年度利用率／事件、CCS 群組、殘差校正、外部來源擴充等不同接法。每列均是該實驗的**獨立預測**；完全相同的 OOF 控制組已去重。數字使用與主表相同的六項指標計算。

### [cdc_clinical_interpretability_results](results/cdc_clinical_interpretability_results/summary.csv)

CDC 健康指標加到醫院臨床特徵，另做單一健康欄位移除／保留控制與 SHAP。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4_Clinical_CDCHealth | 0.235256 | 0.678403 | 0.276939 | 0.243060 | 0.258896 | 0.095640 | 0.233950 |
| Only_DiffWalk | 0.235084 | 0.678039 | 0.277039 | 0.243148 | 0.258990 | 0.095663 | 0.233638 |
| E2_Clinical | 0.234919 | 0.678280 | 0.276334 | 0.242530 | 0.258331 | 0.095664 | 0.233694 |
| E1_Baseline | 0.234825 | 0.676851 | 0.276234 | 0.242442 | 0.258237 | 0.095689 | 0.233800 |
| No_PhysHlth | 0.234816 | 0.677879 | 0.274924 | 0.241292 | 0.257012 | 0.095663 | 0.233639 |
| No_GenHlth | 0.234706 | 0.677794 | 0.276234 | 0.242441 | 0.258237 | 0.095673 | 0.233353 |
| Only_GenHlth | 0.234560 | 0.678319 | 0.277241 | 0.243325 | 0.259178 | 0.095671 | 0.233461 |
| Only_PhysHlth | 0.234338 | 0.677868 | 0.278751 | 0.244651 | 0.260590 | 0.095687 | 0.233243 |
| No_DiffWalk | 0.234110 | 0.677909 | 0.274824 | 0.241204 | 0.256919 | 0.095693 | 0.232991 |

### [cdc_diffwalk_control_results](results/cdc_diffwalk_control_results/summary.csv)

對 CDC 行動困難 DiffWalk 做真實值、置換值及全部健康指標控制。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Permuted_DiffWalk | 0.235028 | 0.678541 | 0.274622 | 0.241026 | 0.256730 | 0.095662 | 0.233750 |

### [meps_augmented_hgb_seed777](results/meps_augmented_hgb_seed777/summary.csv)

把 MEPS 風險特徵直接加到醫院 HGB，對照既有 HGB 和固定融合。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_fixed_50 | 0.237485 | 0.678561 | 0.279456 | 0.245270 | 0.261249 | 0.096527 | 0.231596 |
| hgb_tuned | 0.237247 | 0.676933 | 0.279557 | 0.245358 | 0.261343 | 0.095573 | 0.235987 |
| hgb_meps_augmented | 0.236877 | 0.675689 | 0.277442 | 0.243502 | 0.259366 | 0.095612 | 0.235749 |

### [meps_bridge8_results](results/meps_bridge8_results/summary.csv)

以年齡、性別、疾病等 Bridge8 共享變數接 MEPS ER／住院利用率。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct8_MEPS_Both8 | 0.235585 | 0.678829 | 0.274622 | 0.241027 | 0.256730 | 0.095632 | 0.234141 |
| Direct_Bridge8 | 0.235104 | 0.678881 | 0.276838 | 0.242972 | 0.258802 | 0.095644 | 0.233891 |
| MEPS_IP_8 | 0.234870 | 0.678401 | 0.278147 | 0.244120 | 0.260025 | 0.095655 | 0.233695 |
| MEPS_ER_8 | 0.234828 | 0.678292 | 0.275629 | 0.241911 | 0.257672 | 0.095668 | 0.233689 |
| MEPS_Both_8 | 0.234393 | 0.677959 | 0.276737 | 0.242883 | 0.258707 | 0.095676 | 0.233244 |

### [meps_ccs_group_hgb_seed777](results/meps_ccs_group_hgb_seed777/summary.csv)

把 MEPS 次年利用率依 AHRQ CCS 疾病群組彙整，再加到 HGB。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ccs_meps_group_hgb | 0.238943 | 0.678128 | 0.281672 | 0.247213 | 0.263320 | 0.095482 | 0.237537 |

### [meps_cell_results](results/meps_cell_results/summary.csv)

以人口與健康／利用率 cell 匹配 MEPS 的群組風險。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct_Demog | 0.235160 | 0.679216 | 0.277140 | 0.243237 | 0.259084 | 0.095640 | 0.233749 |
| Cell_All | 0.235131 | 0.677912 | 0.277845 | 0.243855 | 0.259743 | 0.095679 | 0.233655 |
| Cell_Health | 0.234925 | 0.678129 | 0.276737 | 0.242883 | 0.258707 | 0.095669 | 0.233690 |
| Cell_ERIP | 0.234824 | 0.677980 | 0.276435 | 0.242618 | 0.258425 | 0.095669 | 0.233733 |
| Cell_Util | 0.234399 | 0.677707 | 0.277744 | 0.243767 | 0.259649 | 0.095694 | 0.233265 |
| Direct_Demog_Cell_All | 0.234250 | 0.678035 | 0.276234 | 0.242441 | 0.258237 | 0.095705 | 0.233012 |

### [meps_event_ccs_soft_seed777](results/meps_event_ccs_soft_seed777/summary.csv)

以 ICD 前綴及上層群組平滑 MEPS 事件 CCS 訊號，含置換控制。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ccs_meps_soft_hgb | 0.237349 | 0.676970 | 0.280463 | 0.246154 | 0.262191 | 0.095575 | 0.235961 |
| ccs_meps_soft_shuffled | 0.234799 | 0.675568 | 0.278449 | 0.244385 | 0.260307 | 0.095712 | 0.233799 |

### [meps_event_rank_correction_seed777](results/meps_event_rank_correction_seed777/summary.csv)

只對較高風險患者套用 MEPS 事件排名校正，另比較固定與驗證集門控強度。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_ccs_hgb | 0.238865 | 0.677766 | 0.280967 | 0.246596 | 0.262662 | 0.095508 | 0.237699 |
| meps_rank_gated | 0.238734 | 0.677816 | 0.281974 | 0.247480 | 0.263603 | 0.095517 | 0.237446 |
| meps_rank_fixed_025 | 0.238504 | 0.677879 | 0.282779 | 0.248187 | 0.264356 | 0.095511 | 0.237380 |

### [meps_group_prior_hgb_seed777](results/meps_group_prior_hgb_seed777/summary.csv)

比較醫院內部疾病群組先驗與 MEPS 事件群組先驗。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| hospital_group_prior | 0.237010 | 0.677386 | 0.279053 | 0.244916 | 0.260872 | 0.095609 | 0.235537 |
| meps_group_prior | 0.235659 | 0.677995 | 0.278751 | 0.244651 | 0.260590 | 0.095629 | 0.234655 |

### [meps_harmonized_weighted_residual_seed777](results/meps_harmonized_weighted_residual_seed777/summary.csv)

以醫院與 MEPS 分布權重調整外部風險，再校正 HGB 殘差。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| meps_residual | 0.237102 | 0.676994 | 0.279255 | 0.245093 | 0.261061 | 0.095566 | 0.235777 |

### [meps_hgb_results](results/meps_hgb_results/summary.csv)

HGB 基線與加入 MEPS 風險的 HGB。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clinical_HGB | 0.235741 | 0.674921 | 0.276334 | 0.242531 | 0.258331 | 0.095676 | 0.234761 |
| MEPS_HGB | 0.235495 | 0.674701 | 0.277644 | 0.243680 | 0.259555 | 0.095697 | 0.234354 |

### [meps_multitask_results](results/meps_multitask_results/summary.csv)

UCI 與 MEPS 多任務學習、預訓練及醫院 MLP 對照。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MEPS_Multitask | 0.223721 | 0.671323 | 0.267674 | 0.234928 | 0.250234 | 0.096493 | 0.220328 |
| MEPS_Pretrained | 0.222629 | 0.672212 | 0.268379 | 0.235548 | 0.250894 | 0.096483 | 0.218811 |
| UCI_MLP | 0.222066 | 0.670126 | 0.267875 | 0.235106 | 0.250423 | 0.096756 | 0.216965 |

### [meps_nested_hgb_blend_seed777](results/meps_nested_hgb_blend_seed777/summary.csv)

內層驗證挑 HGB、MEPS XGBoost 與融合權重。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_nested | 0.237015 | 0.678132 | 0.278751 | 0.244651 | 0.260590 | 0.096766 | 0.228183 |
| xgb_meps_tuned | 0.234601 | 0.677285 | 0.275932 | 0.242177 | 0.257955 | 0.099983 | 0.219631 |

### [meps_residual_correction_seed777](results/meps_residual_correction_seed777/summary.csv)

用 MEPS 外部風險對醫院 HGB 預測做殘差校正。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| meps_residual | 0.237245 | 0.676987 | 0.279355 | 0.245182 | 0.261155 | 0.095565 | 0.235877 |

### [meps_utilization_results](results/meps_utilization_results/summary.csv)

比較 MEPS 2、6、8 年期 ER／住院利用率訊號。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MEPS_Both_2 | 0.234983 | 0.678269 | 0.276838 | 0.242972 | 0.258802 | 0.095661 | 0.233688 |
| MEPS_ER_6 | 0.234431 | 0.678490 | 0.278046 | 0.244032 | 0.259931 | 0.095670 | 0.233367 |
| MEPS_IP_2 | 0.234324 | 0.678265 | 0.276334 | 0.242530 | 0.258331 | 0.095680 | 0.233215 |
| MEPS_ER_2 | 0.234271 | 0.678457 | 0.277341 | 0.243414 | 0.259272 | 0.095685 | 0.233036 |
| MEPS_IP_6 | 0.234257 | 0.678135 | 0.276234 | 0.242441 | 0.258236 | 0.095686 | 0.233034 |
| MEPS_Both_6 | 0.234199 | 0.677947 | 0.274320 | 0.240762 | 0.256448 | 0.095681 | 0.233061 |

### [real_external_sources_hgb_seed777](results/real_external_sources_hgb_seed777/summary.csv)

擴大 MEPS 年份，並加入 NHANES 糖尿病住院訊號。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ccs_meps_extended_hgb | 0.238625 | 0.676461 | 0.282377 | 0.247833 | 0.263980 | 0.095546 | 0.237371 |
| ccs_meps_nhanes_hgb | 0.238111 | 0.676110 | 0.280463 | 0.246154 | 0.262191 | 0.095577 | 0.236977 |
| ccs_nhanes_hgb | 0.237414 | 0.676730 | 0.278248 | 0.244209 | 0.260119 | 0.095599 | 0.236131 |

## 較早 99,340 筆隊列的完整五折實驗

這些仍是病人互斥五折，主要涵蓋 MEPS 縱向預訓練、風險融合與事件遷移；但隊列少 3 筆且折別與近期主表不同，**只在本節內描述**。全部從完整 OOF 以相同最高 10% 警示規則重算六項指標；不直接拿來和 99,343 筆表判斷勝負。

### [meps_embedding_tuned_results](results/meps_embedding_tuned_results/summary.csv)

調整 MEPS embedding 對醫院 XGBoost 的輸入方式。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| clinical | 0.209449 | 0.651929 | 0.246123 | 0.216014 | 0.230088 | 0.097207 | 0.208555 |
| multitask_tuned | 0.208415 | 0.649419 | 0.246626 | 0.216457 | 0.230559 | 0.097290 | 0.207296 |

### [meps_embedding_xgb_results](results/meps_embedding_xgb_results/summary.csv)

比較 MEPS 預訓練、多任務、隨機與置換 embedding 的 XGBoost。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| clinical | 0.209917 | 0.652887 | 0.246626 | 0.216456 | 0.230559 | 0.097167 | 0.208976 |
| multitask | 0.209831 | 0.651765 | 0.246727 | 0.216545 | 0.230653 | 0.097193 | 0.208744 |
| pretrained | 0.209534 | 0.650341 | 0.246022 | 0.215927 | 0.229994 | 0.097239 | 0.208336 |
| random | 0.209019 | 0.650207 | 0.246928 | 0.216722 | 0.230841 | 0.097265 | 0.207953 |
| shuffled | 0.208424 | 0.649162 | 0.247432 | 0.217164 | 0.231312 | 0.097310 | 0.207299 |

### [meps_event_blend_results](results/meps_event_blend_results/summary.csv)

MEPS 事件風險與臨床 XGBoost 做固定或驗證集融合，含置換控制。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_shuffled | 0.209914 | 0.652886 | 0.246727 | 0.216545 | 0.230653 | 0.097173 | 0.208991 |
| blend_event | 0.209904 | 0.652868 | 0.246928 | 0.216721 | 0.230841 | 0.097720 | 0.204519 |
| blend_random | 0.209852 | 0.652761 | 0.246727 | 0.216545 | 0.230653 | 0.097399 | 0.206142 |
| risk_shuffled | 0.117641 | 0.513900 | 0.118933 | 0.104385 | 0.111185 | 0.100902 | 0.117366 |
| risk_random | 0.115894 | 0.502536 | 0.116818 | 0.102528 | 0.109207 | 0.100920 | 0.115619 |
| risk_event | 0.111573 | 0.494876 | 0.103122 | 0.090506 | 0.096403 | 0.100923 | 0.112712 |

### [meps_event_transfer_results](results/meps_event_transfer_results/summary.csv)

以 MEPS 日期事件預訓練風險網路，再遷移至醫院主模型。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| event_pretrained | 0.208973 | 0.651108 | 0.247130 | 0.216899 | 0.231030 | 0.097233 | 0.208039 |
| event_shuffled | 0.208450 | 0.649511 | 0.247029 | 0.216810 | 0.230935 | 0.097332 | 0.207079 |
| random_risk | 0.208354 | 0.649362 | 0.245217 | 0.215219 | 0.229241 | 0.097327 | 0.207193 |

### [meps_longitudinal_aligned_results](results/meps_longitudinal_aligned_results/summary.csv)

MEPS 相鄰年度對齊的預訓練／多任務編碼器。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| aligned_pretrained | 0.201978 | 0.644105 | 0.239275 | 0.210004 | 0.223686 | 0.097624 | 0.201336 |
| aligned_mtl | 0.201502 | 0.643669 | 0.240483 | 0.211065 | 0.224816 | 0.097639 | 0.201378 |
| aligned_shuffled_mtl | 0.201338 | 0.644054 | 0.242800 | 0.213098 | 0.226981 | 0.097648 | 0.199808 |
| uci_only | 0.200808 | 0.643094 | 0.241893 | 0.212303 | 0.226134 | 0.097686 | 0.198996 |

### [meps_longitudinal_blend_2008_2016_results](results/meps_longitudinal_blend_2008_2016_results/summary.csv)

把縱向融合使用的 MEPS 年份擴至 2016。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.210137 | 0.653321 | 0.247533 | 0.217252 | 0.231406 | 0.097168 | 0.208953 |
| blend_shuffled_mtl | 0.209495 | 0.652127 | 0.247633 | 0.217340 | 0.231500 | 0.097223 | 0.208337 |
| risk_shuffled_mtl | 0.207694 | 0.648863 | 0.246022 | 0.215926 | 0.229994 | 0.097359 | 0.206169 |
| risk_mtl | 0.205451 | 0.646798 | 0.244612 | 0.214688 | 0.228675 | 0.097748 | 0.200624 |

### [meps_longitudinal_blend_2008_2024_results](results/meps_longitudinal_blend_2008_2024_results/summary.csv)

把縱向融合使用的 MEPS 年份擴至 2024。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.210002 | 0.653402 | 0.248338 | 0.217959 | 0.232159 | 0.097150 | 0.209156 |
| blend_shuffled_mtl | 0.209985 | 0.652837 | 0.247231 | 0.216986 | 0.231123 | 0.097177 | 0.208927 |
| risk_mtl | 0.209245 | 0.651656 | 0.245217 | 0.215219 | 0.229241 | 0.097190 | 0.208349 |
| risk_shuffled_mtl | 0.208178 | 0.649793 | 0.245217 | 0.215219 | 0.229241 | 0.097291 | 0.206947 |

### [meps_longitudinal_blend_ensemble_results](results/meps_longitudinal_blend_ensemble_results/summary.csv)

將多個縱向風險模型的 OOF 融合。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.210191 | 0.652903 | 0.248640 | 0.218224 | 0.232441 | 0.097173 | 0.209136 |
| blend_shuffled_mtl | 0.210061 | 0.652006 | 0.247533 | 0.217252 | 0.231406 | 0.097214 | 0.208784 |
| clinical | 0.210040 | 0.652528 | 0.247633 | 0.217340 | 0.231500 | 0.097187 | 0.208949 |
| risk_mtl | 0.209857 | 0.651497 | 0.247734 | 0.217429 | 0.231594 | 0.097197 | 0.208792 |
| risk_shuffled_mtl | 0.209752 | 0.650901 | 0.245821 | 0.215750 | 0.229806 | 0.097252 | 0.208488 |

### [meps_longitudinal_blend_results](results/meps_longitudinal_blend_results/summary.csv)

2008–2012 MEPS 縱向多任務風險與醫院 XGBoost 融合，含置換控制。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.210155 | 0.653002 | 0.246727 | 0.216545 | 0.230653 | 0.097154 | 0.209250 |
| risk_mtl | 0.209609 | 0.650839 | 0.245116 | 0.215130 | 0.229146 | 0.097209 | 0.208497 |
| blend_shuffled_mtl | 0.209519 | 0.651896 | 0.245921 | 0.215837 | 0.229899 | 0.097278 | 0.207807 |
| risk_shuffled_mtl | 0.206810 | 0.647995 | 0.244411 | 0.214513 | 0.228488 | 0.097502 | 0.204225 |

### [meps_longitudinal_blend_seed2024](results/meps_longitudinal_blend_seed2024/summary.csv)

同一 99,340 筆五折上換縱向模型隨機 seed。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.210013 | 0.652869 | 0.246526 | 0.216367 | 0.230464 | 0.097177 | 0.208916 |
| clinical | 0.209891 | 0.652392 | 0.246828 | 0.216633 | 0.230747 | 0.097189 | 0.208758 |
| blend_shuffled_mtl | 0.209848 | 0.651703 | 0.248439 | 0.218047 | 0.232253 | 0.097211 | 0.208684 |
| risk_shuffled_mtl | 0.209312 | 0.650094 | 0.247835 | 0.217518 | 0.231689 | 0.097248 | 0.208115 |
| risk_mtl | 0.209046 | 0.650944 | 0.247130 | 0.216898 | 0.231029 | 0.097221 | 0.208024 |

### [meps_longitudinal_blend_seed3407](results/meps_longitudinal_blend_seed3407/summary.csv)

同一 99,340 筆五折上換縱向模型隨機 seed。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_shuffled_mtl | 0.209539 | 0.651316 | 0.247331 | 0.217076 | 0.231218 | 0.097225 | 0.208426 |
| risk_shuffled_mtl | 0.209343 | 0.651117 | 0.247633 | 0.217341 | 0.231500 | 0.097218 | 0.208417 |
| blend_mtl | 0.209302 | 0.651178 | 0.246526 | 0.216369 | 0.230465 | 0.097253 | 0.208108 |
| clinical | 0.209068 | 0.650641 | 0.245619 | 0.215573 | 0.229617 | 0.097268 | 0.207895 |
| risk_mtl | 0.208859 | 0.649754 | 0.246928 | 0.216722 | 0.230841 | 0.097262 | 0.207764 |

### [meps_longitudinal_results](results/meps_longitudinal_results/summary.csv)

早期 MEPS 縱向共享編碼器、預訓練及多任務模型的完整五折。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| longitudinal_mtl | 0.201473 | 0.643687 | 0.241793 | 0.212214 | 0.226040 | 0.097655 | 0.199854 |
| direct_shared | 0.201459 | 0.643621 | 0.242598 | 0.212921 | 0.226793 | 0.097655 | 0.200067 |
| pretrained | 0.201408 | 0.643759 | 0.239376 | 0.210093 | 0.223781 | 0.097647 | 0.199983 |
| shuffled_mtl | 0.201383 | 0.643849 | 0.241490 | 0.211949 | 0.225758 | 0.097650 | 0.200117 |

### [meps_longitudinal_risk_xgb_results](results/meps_longitudinal_risk_xgb_results/summary.csv)

將 MEPS 縱向預訓練風險直接作 XGBoost 輸入。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk_shuffled | 0.209443 | 0.650575 | 0.245720 | 0.215662 | 0.229712 | 0.097235 | 0.208436 |
| risk_mtl | 0.209375 | 0.650451 | 0.245619 | 0.215572 | 0.229617 | 0.097228 | 0.208295 |
| risk_pretrained | 0.208722 | 0.649887 | 0.246123 | 0.216015 | 0.230088 | 0.097281 | 0.207747 |
| risk_shuffled_mtl | 0.207466 | 0.648459 | 0.246224 | 0.216103 | 0.230182 | 0.097329 | 0.206574 |

### [meps_longitudinal_tuned_blend_results](results/meps_longitudinal_tuned_blend_results/summary.csv)

以驗證資料調整縱向風險與臨床預測的融合權重。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_mtl | 0.209894 | 0.653195 | 0.246425 | 0.216280 | 0.230370 | 0.097156 | 0.208990 |
| blend_shuffled_mtl | 0.209557 | 0.651815 | 0.245821 | 0.215749 | 0.229805 | 0.097280 | 0.207812 |
| risk_mtl | 0.209539 | 0.651656 | 0.245317 | 0.215307 | 0.229335 | 0.097175 | 0.208874 |
| risk_shuffled_mtl | 0.207254 | 0.648163 | 0.243807 | 0.213982 | 0.227923 | 0.097487 | 0.204788 |

### [meps_longitudinal_xgb_results](results/meps_longitudinal_xgb_results/summary.csv)

比較 MEPS 對齊預訓練／多任務 embedding 與 XGBoost。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.209416 | 0.650661 | 0.247029 | 0.216810 | 0.230935 | 0.097225 | 0.208403 |
| aligned_mtl | 0.209080 | 0.652012 | 0.244814 | 0.214866 | 0.228864 | 0.097183 | 0.208129 |
| aligned_pretrained | 0.208722 | 0.652186 | 0.243001 | 0.213275 | 0.227170 | 0.097182 | 0.207954 |
| aligned_shuffled | 0.207800 | 0.649506 | 0.243202 | 0.213452 | 0.227358 | 0.097308 | 0.206971 |
| aligned_shuffled_mtl | 0.207357 | 0.648704 | 0.245015 | 0.215042 | 0.229052 | 0.097348 | 0.206178 |

### [meps_multisource_blend_results](results/meps_multisource_blend_results/summary.csv)

合併 MEPS 年度縱向與住院事件外部任務。

| Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend_shuffled | 0.209821 | 0.652792 | 0.249547 | 0.219019 | 0.233289 | 0.097279 | 0.207551 |
| blend_event | 0.209600 | 0.652758 | 0.248036 | 0.217694 | 0.231877 | 0.097179 | 0.208616 |
| risk_event | 0.208858 | 0.651500 | 0.246626 | 0.216457 | 0.230559 | 0.097208 | 0.207954 |
| risk_shuffled | 0.206571 | 0.647581 | 0.243807 | 0.213982 | 0.227923 | 0.097564 | 0.202893 |

## 其他 seed 與跨 seed OOF 平均

以下每個 seed 各自完成病人互斥五折，卻不使用 seed 777 的相同測試折；跨 seed 平均則是逐筆多個 OOF 預測的平均。用來檢查穩定性，不併入主表排名。

### 99,343 筆・其他 seed

| Experiment | Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [meps_bridge8_seed2024](results/meps_bridge8_seed2024/summary.csv) | Direct8_MEPS_Both8 | 0.233741 | 0.677252 | 0.273964 | 0.240500 | 0.256144 | 0.095738 | 0.232652 |
| [meps_bridge8_seed2024](results/meps_bridge8_seed2024/summary.csv) | Clinical | 0.232897 | 0.677103 | 0.274668 | 0.241118 | 0.256802 | 0.095777 | 0.231770 |
| [meps_bridge8_seed3407](results/meps_bridge8_seed3407/summary.csv) | Direct8_MEPS_Both8 | 0.234106 | 0.677791 | 0.276709 | 0.242884 | 0.258696 | 0.095676 | 0.233724 |
| [meps_bridge8_seed3407](results/meps_bridge8_seed3407/summary.csv) | Clinical | 0.233321 | 0.677429 | 0.277213 | 0.243326 | 0.259166 | 0.095716 | 0.232925 |
| [meps_hgb_seed2024](results/meps_hgb_seed2024/summary.csv) | MEPS_HGB | 0.235867 | 0.673560 | 0.273764 | 0.240323 | 0.255956 | 0.095705 | 0.234758 |
| [meps_hgb_seed2024](results/meps_hgb_seed2024/summary.csv) | Clinical_HGB | 0.235085 | 0.674264 | 0.275677 | 0.242002 | 0.257744 | 0.095719 | 0.234196 |
| [meps_hgb_seed3407](results/meps_hgb_seed3407/summary.csv) | Clinical_HGB | 0.234257 | 0.673408 | 0.275199 | 0.241559 | 0.257284 | 0.095724 | 0.233925 |
| [meps_hgb_seed3407](results/meps_hgb_seed3407/summary.csv) | MEPS_HGB | 0.232447 | 0.672528 | 0.272883 | 0.239526 | 0.255119 | 0.095811 | 0.231969 |

### 跨 seed OOF 平均

| Experiment | Model | AP | AUROC | Precision | Recall | F1 | Brier | Pooled_AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [meps_hgb_xgb_ensemble](results/meps_hgb_xgb_ensemble/summary.csv) | external | 0.239347 | 0.679698 | 0.278348 | 0.244297 | 0.260213 | 0.095458 | 0.238312 |
| [meps_hgb_xgb_ensemble](results/meps_hgb_xgb_ensemble/summary.csv) | hgb | 0.239282 | 0.677475 | 0.280564 | 0.246242 | 0.262285 | 0.095497 | 0.238323 |
| [meps_hgb_xgb_ensemble](results/meps_hgb_xgb_ensemble/summary.csv) | no_external | 0.238814 | 0.679395 | 0.279053 | 0.244916 | 0.260872 | 0.095484 | 0.237798 |
| [meps_hgb_xgb_ensemble](results/meps_hgb_xgb_ensemble/summary.csv) | meps_xgb | 0.236293 | 0.679227 | 0.277442 | 0.243502 | 0.259367 | 0.095597 | 0.235077 |
| [meps_hgb_xgb_ensemble](results/meps_hgb_xgb_ensemble/summary.csv) | clinical_xgb | 0.235498 | 0.678852 | 0.277442 | 0.243502 | 0.259366 | 0.095636 | 0.234410 |

## CDC 摘要型五折實驗

這些來源保存五折六項指標，但沒有可逐筆重算全部模型的 OOF；因此採原摘要的驗證集選門檻，Precision／Recall／F1 與上表的固定最高 10% 不同。重複的完全相同摘要行已去重。

### [cdc_clinical_results](results/cdc_clinical_results/summary_mean_std.csv)

CDC 健康與慢性病特徵加入醫院臨床模型，並與臨床基線比較。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E4_Clinical_CDCHealth | 0.235256 | 0.678403 | 0.211719 | 0.465705 | 0.290461 | 0.095640 |
| E2_Clinical | 0.234919 | 0.678280 | 0.221075 | 0.425836 | 0.290581 | 0.095664 |
| E1_Baseline | 0.234825 | 0.676851 | 0.214922 | 0.452978 | 0.289849 | 0.095689 |
| E3_Clinical_CDCScore | 0.234635 | 0.678024 | 0.212749 | 0.460127 | 0.289241 | 0.095671 |

### [cdc_comorbidity_results](results/cdc_comorbidity_results/summary_mean_std.csv)

將 CDC 共病相關資訊與醫院診斷特徵結合。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E6_Health | 0.234924 | 0.677350 | 0.214946 | 0.455727 | 0.291403 | 0.095673 |
| E3_Bridge6 | 0.234779 | 0.677566 | 0.213902 | 0.458635 | 0.289853 | 0.095673 |
| E2_Bridge4 | 0.234671 | 0.677202 | 0.214929 | 0.452790 | 0.289653 | 0.095686 |
| E5_PhysHlth | 0.234476 | 0.677225 | 0.221577 | 0.420624 | 0.288050 | 0.095691 |
| E4_GenHlth | 0.234414 | 0.677361 | 0.213545 | 0.453423 | 0.289086 | 0.095694 |
| E7_Socio | 0.234313 | 0.676898 | 0.218781 | 0.436799 | 0.290237 | 0.095710 |
| E8_All_CDC | 0.234143 | 0.676969 | 0.218593 | 0.434771 | 0.289871 | 0.095703 |

### [cdc_diagnosis_aware_results](results/cdc_diagnosis_aware_results/summary_mean_std.csv)

以醫院診斷分組對齊 CDC 外部健康訊號。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E5_NaiveHealth | 0.235246 | 0.676976 | 0.217931 | 0.435999 | 0.288823 | 0.095664 |
| E6_DiagnosisAware | 0.235090 | 0.677077 | 0.219680 | 0.438483 | 0.290815 | 0.095666 |
| E4_Bridge6 | 0.234877 | 0.677508 | 0.219591 | 0.437157 | 0.291282 | 0.095679 |
| E7_AwarePlus | 0.234401 | 0.677100 | 0.213853 | 0.455979 | 0.289450 | 0.095691 |
| E2_Reliable4 | 0.234401 | 0.677212 | 0.213044 | 0.461195 | 0.290165 | 0.095693 |
| E3_ReliableHealth | 0.234343 | 0.677106 | 0.217920 | 0.433615 | 0.288423 | 0.095688 |

### [cdc_distribution_alignment_results](results/cdc_distribution_alignment_results/summary_mean_std.csv)

對 CDC 與醫院資料的分布差異做對齊，再比較主要任務表現。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E4_Aligned_Health | 0.234849 | 0.677049 | 0.219511 | 0.427601 | 0.287714 | 0.095684 |
| E5_Aligned_All | 0.233934 | 0.677015 | 0.217869 | 0.432914 | 0.288858 | 0.095721 |

### [cdc_encoder_control_results](results/cdc_encoder_control_results/summary_mean_std.csv)

比較 CDC 編碼器、置換控制與僅使用醫院資料的模型。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Bridge | 0.232240 | 0.676257 | 0.224300 | 0.410454 | 0.286932 | 0.095789 |
| RandomEncoder | 0.232117 | 0.676289 | 0.217220 | 0.437148 | 0.286181 | 0.095794 |
| CDCPretrained | 0.232009 | 0.675971 | 0.217232 | 0.429993 | 0.287444 | 0.095807 |
| RandomEncoder/ClassWeight2 | 0.231633 | 0.676147 | 0.213808 | 0.448465 | 0.287395 | 0.103313 |
| Bridge/ClassWeight2 | 0.231605 | 0.676319 | 0.213830 | 0.454222 | 0.288439 | 0.103337 |
| CDCPretrained/ClassWeight2 | 0.231487 | 0.676223 | 0.219118 | 0.438039 | 0.291067 | 0.103309 |
| Bridge/ClassWeight4 | 0.230525 | 0.676179 | 0.214911 | 0.453075 | 0.289343 | 0.139456 |
| CDCPretrained/ClassWeight4 | 0.230378 | 0.676047 | 0.216691 | 0.445029 | 0.289674 | 0.139361 |
| RandomEncoder/ClassWeight4 | 0.230209 | 0.675919 | 0.214256 | 0.448111 | 0.288047 | 0.139401 |
| RandomEncoder/SMOTENC | 0.222964 | 0.666958 | 0.207860 | 0.446791 | 0.283401 | 0.096536 |
| Bridge/SMOTENC | 0.221204 | 0.665722 | 0.209831 | 0.436978 | 0.282771 | 0.096629 |
| CDCPretrained/SMOTENC | 0.220695 | 0.664874 | 0.207098 | 0.439357 | 0.279347 | 0.096726 |

### [cdc_encoder_cv_results](results/cdc_encoder_cv_results/summary_mean_std.csv)

以五折驗證 CDC 編碼器轉移與臨床模型。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E4_CDC_Combined | 0.234186 | 0.677256 | 0.218143 | 0.432205 | 0.289230 | 0.100255 |
| E3_CDC_Embedding | 0.233726 | 0.677213 | 0.216956 | 0.439103 | 0.288280 | 0.097233 |
| E2_CDC_Scores | 0.233266 | 0.676688 | 0.220614 | 0.421770 | 0.285793 | 0.114819 |
| E1_Bridge | 0.232902 | 0.676819 | 0.214281 | 0.448821 | 0.288059 | 0.114814 |

### [cdc_ih_5fold_results](results/cdc_ih_5fold_results/summary_mean_std.csv)

比較 CDC 外部健康訊號在不同醫院特徵集合下的效果。

| Model | AP | AUROC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CDC_All | 0.234544 | 0.677255 | 0.214605 | 0.457131 | 0.287679 | 0.095692 |
| CDC_Multi | 0.233780 | 0.676580 | 0.218934 | 0.429640 | 0.288136 | 0.095735 |
| Bridge/RUS | 0.232719 | 0.676112 | 0.210356 | 0.460843 | 0.287549 | 0.108634 |
| CDC_Multi/RUS | 0.232289 | 0.676210 | 0.219233 | 0.430358 | 0.289529 | 0.108659 |
| CDC_All/RUS | 0.232153 | 0.675931 | 0.219043 | 0.426210 | 0.287698 | 0.108664 |
| Bridge/ROS | 0.232105 | 0.675420 | 0.218098 | 0.432997 | 0.288971 | 0.107942 |
| Baseline/RUS | 0.231955 | 0.675656 | 0.219808 | 0.427090 | 0.288580 | 0.108684 |
| Baseline/ROS | 0.231277 | 0.675167 | 0.221009 | 0.415242 | 0.288012 | 0.108016 |
| CDC_All/ROS | 0.231259 | 0.675112 | 0.216106 | 0.436093 | 0.287846 | 0.107992 |
| CDC_Multi/ROS | 0.231051 | 0.675239 | 0.217774 | 0.424953 | 0.286420 | 0.107962 |
| Bridge/SMOTE | 0.230381 | 0.675099 | 0.213319 | 0.443169 | 0.285579 | 0.095950 |
| CDC_Multi/SMOTE | 0.230097 | 0.674872 | 0.214639 | 0.436891 | 0.286648 | 0.095970 |
| CDC_All/SMOTE | 0.229963 | 0.674739 | 0.209110 | 0.457760 | 0.285165 | 0.095981 |
| CDC_All/ClassWeight | 0.229947 | 0.675418 | 0.214401 | 0.437510 | 0.287162 | 0.218568 |
| CDC_Multi/ClassWeight | 0.229862 | 0.675775 | 0.222886 | 0.415747 | 0.286475 | 0.218341 |
| Baseline/SMOTE | 0.229811 | 0.674725 | 0.217530 | 0.426017 | 0.286619 | 0.095992 |
| Bridge/ClassWeight | 0.229588 | 0.675832 | 0.223126 | 0.410017 | 0.288228 | 0.218550 |
| Baseline/ClassWeight | 0.229549 | 0.675616 | 0.216186 | 0.437237 | 0.287906 | 0.218665 |

## 重複結果的來源

下列實驗已完成五折，但其保存的預測與上表某個模型完全一致；為避免同一模型重複算入，僅保留較前面的主表來源：

- [meps_event_ccs_group_hgb_seed777](results/meps_event_ccs_group_hgb_seed777/summary.csv)：對應 `meps_augmented_hgb_seed777/p_ensemble_fixed_50, meps_augmented_hgb_seed777/p_hgb_tuned, paper_fair_comparison_seed777/ccs_meps_group_hgb, paper_fair_comparison_seed777/ccs_only_hgb`。

## SHAP：模型主要依據什麼預測？

五個外層測試折各抽 1,000 筆，CatBoost 使用 TreeSHAP；HGB 的 one-hot 歸因加總回原始欄位。數值單位為模型 **raw log-odds**，排序依平均絕對 SHAP；它表示預測受該欄位影響的平均幅度，不表示風險方向或因果效果。[完整全域值](results/paper_fair_comparison_seed777/shap_global.csv)、[各折值](results/paper_fair_comparison_seed777/shap_fold.csv)。

| Model | Rank | Feature | 平均絕對 SHAP（log-odds） |
|---|---|---|---:|
| catboost_family | 1 | number_inpatient | 0.274974 |
| catboost_family | 2 | discharge_disposition_id | 0.145840 |
| catboost_family | 3 | number_diagnoses | 0.051209 |
| catboost_family | 4 | payer_code | 0.047029 |
| catboost_family | 5 | diag_1 | 0.046731 |
| catboost_raw | 1 | number_inpatient | 0.238927 |
| catboost_raw | 2 | discharge_disposition_id | 0.139130 |
| catboost_raw | 3 | diag_1 | 0.077924 |
| catboost_raw | 4 | diag_3 | 0.059604 |
| catboost_raw | 5 | payer_code | 0.042366 |
| ccs_meps_refit_hgb | 1 | number_inpatient | 0.290321 |
| ccs_meps_refit_hgb | 2 | discharge_disposition_id | 0.205978 |
| ccs_meps_refit_hgb | 3 | number_diagnoses | 0.046228 |
| ccs_meps_refit_hgb | 4 | diabetesMed | 0.040975 |
| ccs_meps_refit_hgb | 5 | meps_ccs_revisit_mean | 0.040843 |

三個模型的第一、二名都是過去住院次數 `number_inpatient` 與出院去向 `discharge_disposition_id`。原始 ICD CatBoost 的 `diag_1`、`diag_3` 排第三、四，顯示診斷表示有影響；MEPS HGB 的 `meps_ccs_revisit_mean` 排第五、平均絕對 SHAP 0.040843。所有 `meps_` 特徵合計約占該模型平均絕對 SHAP 的 **6.1%**，說明模型確實用到外部訊號，但配對 AP 沒有證明其帶來最終改善。`MeanSHAP_logodds` 的正負平均可能互相抵銷，不能用全域平均判斷單一特徵增加時風險的方向。

[MEPS 30 天共同訓練的 CatBoost SHAP](results/meps_30d_joint_transfer_seed777/shap_global.csv)另對每折 500 筆、共 2,500 筆測試就醫解釋：過往住院次數 0.230564、出院去向 0.146597 仍居前二；`meps_30d_teacher_risk` 為 0.032326、排第 8，`meps_ccs_log_support` 為 0.019019、排第 15。這同樣證明訊號被使用，**不等於**加入外部資料提高 AP。CDC 的[診斷感知 SHAP](results/cdc_clinical_interpretability_results/shap_importance.csv)亦以住院次數及出院去向為主；[DiffWalk 控制](results/cdc_diffwalk_control_results/cdc_shap_summary.csv)有外部特徵歸因，但不單靠歸因宣稱提升。

SHAP 只針對上述已保存解釋檔的模型分析；其他表格中的模型沒有可核對的 SHAP 結果，不推測其特徵排序。

## 統計解讀與報告界線

在同一測試折比較模型時，優先看配對病人 bootstrap；單看五折平均或五折標準差不能證明差異。主表結果經多次嘗試與選模，仍有探索性選模偏差。真實外部資料的成功接入可由特徵來源與 SHAP 證明，但若和同架構、同切分、無外部資料對照的 AP 差沒有穩定正值，就不能宣稱外部資料提升主要任務。本文只報已完成的五折；不填補未產生的數字。

本附錄可由 `python scripts/experiments/build_results_analysis.py` 依目前保存的 OOF 和 CDC 摘要重新生成；重算的逐方法六指標另存於 `results/fivefold_metrics_recomputed.csv`。
