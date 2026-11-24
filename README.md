# 專題

糖尿病 30 天再住院預測

本專題以 UCI 糖尿病住院資料預測出院後 30 天內再住院，並檢驗真實外部資料能否改善預測。主要指標是 **PR-AUC（AP）**：程式使用 `sklearn.metrics.average_precision_score`，與以梯形法積分 PR 曲線的數值可能不同。AUROC、F1、Brier 為輔助指標。以下數字均取自 `results/` 或 `logs/` 中已完成的執行紀錄；尚未執行的程式不列為結果。完整方法、所有已保存指標、論文復現、SHAP 與比較限制見 [實驗結果總報告](RESULTS_ANALYSIS.md)。

## 目前結論

- **同一組 seed 777、五折 patient-disjoint 評估的最高平均 AP 是 CCS-only HGB＋原始 ICD-9 CatBoost 的 0.247374**；舊 HGB 家族最高為 CCS-only HGB 的 0.239179。[新同切分比較](results/paper_fair_comparison_seed777/summary.csv)｜[舊 HGB 每折結果](results/meps_ccs_group_hgb_seed777/fold_results.csv)
- **有使用外部病人資料的最高同組平均 AP 約 0.247140**，由 MEPS event-CCS HGB＋CatBoost 達成。它比相同 CatBoost 的 CCS-only 融合低約 0.0002，配對病人 bootstrap CI 跨零；尚無 MEPS 額外提升 AP 的證據。[結果](results/paper_fair_comparison_seed777/summary.csv)｜[配對比較](results/paper_fair_comparison_seed777/paired_ap_bootstrap.csv)
- 跨三個 seed 的 HGB＋XGBoost 機率平均實驗中，**有 MEPS** 的 pooled OOF AP 為 0.238312，**無外部資料** 為 0.237798，差 +0.000514（病人 bootstrap 95% CI +0.000108 至 +0.000924）。這是不同於上項的「先跨 seed 平均預測，再計算 pooled AP」，不能與五折平均 AP 混排。[結果](results/meps_hgb_xgb_ensemble/summary.csv)｜[配對比較](results/meps_hgb_xgb_ensemble/patient_bootstrap.csv)
- 使用者提供的 Rajpoot (2026) 圖片報告 **20% held-out test AP 0.237**，樣本為 99,340 encounters／69,987 patients；本專題最新五折資料為 99,343／69,990。多出的三筆恰是 `gender=Unknown/Invalid`，但無法單憑圖片確認對方是否因此排除。**分割與樣本數不同，不能以 0.239179 > 0.237 宣稱公平超越 SOTA。**

## 資料與評估口徑

| 資料／資源 | 在專題中的用途 | 注意 |
|---|---|---|
| UCI 糖尿病住院資料 | 主要預測資料；五折流程使用 99,343 筆、69,990 位病人，30 天再住院率約 11.39% | 同一病人不能跨訓練與測試折 |
| MEPS | **真實外部病人資料**；年度追蹤、住院／急診事件與疾病紀錄 | 2008–2024 年度表有 45,436 person-years、21,176 個連續年度配對；精確事件日期的 30 天標籤目前限 2008–2012 |
| CDC BRFSS | 真實外部調查資料，用來估計人口群健康風險或訓練編碼器 | 沒有 UCI 的 30 天再住院標籤 |
| NHANES | 真實外部調查資料，1,747 位自述糖尿病受訪者 | 結果是「過去一年曾住院」，不是 30 天再住院 |
| AHRQ CCS | ICD 診斷的公開分級／對照表 | **不是外部病人 dataset**；CCS-only 不符合「加入外部病人資料」的要求 |

主要分析固定外層 **5-fold stratified patient-disjoint**，最新 HGB／MEPS 比較以 seed 777 為主。外層每折約 79,474 筆訓練、19,869 筆測試；超參數、融合權重或門檻應只用訓練側資料選取。下表的「五折平均 AP」是各折 AP 的算術平均；「pooled OOF AP」是把所有測試折預測合併後再算一次 AP；兩者不相等。舊的固定 holdout 是 train／validation／test 69,613／14,779／14,948 筆，亦不可與五折結果直接比較。

本專題還保留兩套特徵流程：早期較保守的 15 個衍生臨床特徵流程，其 Clinical XGBoost 約 **0.209917**；後期的出院時 46 個原始欄位及延伸臨床特徵流程，Clinical XGBoost 約 **0.234919**。兩套的絕對 AP 不代表同一模型的前後進步。

## 早期固定 holdout：LR、RF、XGBoost、類別不平衡與外部資料

LR、RF、XGBoost 當時使用**相同的單次固定 holdout 分割**，尚未採用五折；這項 LR 分割資訊由使用者確認。此處的 test AP 不與後來的五折平均 AP 混排。RF 和 XGBoost 的 AP 不受 0.5／驗證集 F1 門檻影響，因此每個模型只列一次。[RF 原始指標](results/rf_results/rf_metrics.csv)｜[XGBoost 原始指標](results/xgboost_results/xgb_metrics.csv)

| 方法 | Test AP | Test AUROC | 來源 |
|---|---:|---:|---|
| Logistic Regression (LR) | **0.2194** | 0.6461 | 使用者提供的早期結果；已確認與 RF／XGBoost 同一 holdout 分割，沒有保存 log，前處理細節無法由檔案核對 |
| Random Forest 標準權重 | 0.219298 | 0.647946 | [RF](results/rf_results/rf_metrics.csv) |
| Random Forest `class_weight=balanced` | 0.205252 | 0.649289 | [RF](results/rf_results/rf_metrics.csv) |
| XGBoost 標準權重 | **0.229119** | 0.655925 | [XGBoost](results/xgboost_results/xgb_metrics.csv) |
| XGBoost 類別加權 | 0.218236 | 0.641720 | [XGBoost](results/xgboost_results/xgb_metrics.csv) |
| XGBoost＋CCS（僅精確代碼） | 0.229160 | 0.653881 | [CCS](results/xgboost_ccs_results/metrics.csv) |
| XGBoost＋CCS（含唯一 ICD 前綴） | 0.229820 | 0.658246 | [CCS 前綴](results/xgboost_ccs_prefix_results/metrics.csv)；取代上列為較完整的 CCS 對照方式 |
| XGBoost＋CDC 群組統計 | 0.227831 | 0.653286 | [CDC](results/xgboost_cdc_results/metrics.csv) |
| XGBoost＋CDC＋CCS 前綴 | 0.229389 | 0.654592 | [CDC＋CCS](results/xgboost_cdc_ccs_prefix_results/metrics.csv) |
| XGBoost＋SMOTENC（少數／多數比 0.20） | 0.224612 | 0.653477 | [SMOTENC](results/xgboost_smote_0.20/xgb_metrics.csv) |
| XGBoost＋SMOTENC（0.35／0.50／1.00） | 0.213285／0.201799／0.194979 | 0.639436／0.628686／0.612077 | [0.35](results/xgboost_smote_0.35/xgb_metrics.csv)｜[0.50](results/xgboost_smote_0.50/xgb_metrics.csv)｜[1.00](results/xgboost_smote_1.00/xgb_metrics.csv) |

下表補齊 **XGBoost class weight 與各個 SMOTENC 比例的 test 全部六項指標**。Precision、Recall、F1 使用在 validation 選出的 F1 最佳門檻，再固定用於 test；AP、AUROC、Brier 不受分類門檻影響。這些是早期**單次固定 holdout**，不能與後期五折平均值直接比較。

| 方法 | Test AP／PR-AUC | AUROC | Precision | Recall | F1 | Brier | Validation 選出門檻 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost 標準權重 | 0.229119 | 0.655925 | 0.206793 | 0.419911 | 0.277115 | 0.101391 | 0.131729 |
| XGBoost class weight | 0.218236 | 0.641720 | 0.189797 | 0.488320 | 0.273350 | 0.242900 | 0.507230 |
| XGBoost＋SMOTENC 0.20 | 0.224612 | 0.653477 | 0.213095 | 0.383760 | 0.274027 | 0.101636 | 0.150355 |
| XGBoost＋SMOTENC 0.35 | 0.213285 | 0.639436 | 0.199886 | 0.390990 | 0.264534 | 0.104107 | 0.189829 |
| XGBoost＋SMOTENC 0.50 | 0.201799 | 0.628686 | 0.167984 | 0.508343 | 0.252521 | 0.108397 | 0.199494 |
| XGBoost＋SMOTENC 1.00 | 0.194979 | 0.612077 | 0.169000 | 0.469967 | 0.248603 | 0.132120 | 0.311395 |

來源：[XGBoost 指標](results/xgboost_results/xgb_metrics.csv)｜[SMOTENC 0.20](results/xgboost_smote_0.20/xgb_metrics.csv)｜[0.35](results/xgboost_smote_0.35/xgb_metrics.csv)｜[0.50](results/xgboost_smote_0.50/xgb_metrics.csv)｜[1.00](results/xgboost_smote_1.00/xgb_metrics.csv)。原 CSV 另保留 0.5 門檻、validation 指標及混淆矩陣。

LR 的其他使用者提供指標為 Precision **0.1818**、Recall **0.5072**、F1 **0.2677**、Brier **0.1020**。這組 LR 數字沒有可核對的 log／預測檔；早期 `logs/rf.log`、`logs/xgboost.log` 則與上述 RF／XGBoost CSV 相符。三者可在相同 holdout 分割下描述，但不能從現存紀錄證明 LR 的前處理和模型選擇流程也完全一致。後來的五折類別不平衡比較也沒有改善：無重採樣基線 AP **0.234825**，ROS **0.231277**、RUS **0.231955**、SMOTE **0.229811**、類別加權 **0.229549**。[完整 20 組設定](results/cdc_ih_5fold_results/summary_mean_std.csv)

## 五折出院時流程：CDC、MEPS 與模型比較

下列均為五折**平均 AP**，除非表格特別標明。重複的基線只列在同一方法家族內供配對，不應跨不同腳本直接推論微小差異。完整逐折值位於各來源目錄的 `fold_results.csv`。

### CDC 與早期 MEPS 表徵

| 方法／對照 | 平均 AP | 解讀與來源 |
|---|---:|---|
| 原始出院特徵 XGBoost | 0.234825 | [CDC 臨床比較](results/cdc_clinical_results/summary_mean_std.csv)的 `E1_Baseline` |
| 加入衍生臨床特徵 | 0.234919 | 同上 `E2_Clinical`；亦為多個 MEPS 實驗的 Clinical 基線 |
| CDC 健康統計 | **0.235256** | 同上 `E4_Clinical_CDCHealth`；相對 Clinical +0.000337 |
| CDC 疾病對齊／診斷感知 | 0.235090 | [診斷感知](results/cdc_diagnosis_aware_results/summary_mean_std.csv) `E6_DiagnosisAware`；其他可靠性／社經變體見 CSV |
| CDC 分布對齊 | 0.234849 | [分布對齊](results/cdc_distribution_alignment_results/summary_mean_std.csv) `E4_Aligned_Health`，未超過原 CDC 健康統計 |
| CDC DiffWalk 真／置換對照 | 0.235084／0.235028 | [置換對照](results/cdc_diffwalk_control_results/summary.csv)；微小差距不足以歸因於真實外部訊號 |
| CDC 表徵＋群組分數 | 0.234186 | [編碼器比較](results/cdc_encoder_cv_results/summary_mean_std.csv)；該腳本 Bridge 基線 0.232902 |
| MEPS 人口特徵直接加入 | 0.235104 | [Bridge8](results/meps_bridge8_results/summary.csv) `Direct_Bridge8` |
| MEPS ER/IP 風險＋人口特徵 | **0.235585** | 同上 `Direct8_MEPS_Both8`，對 Clinical +0.000666；單 seed 病人 bootstrap 區間跨零 |
| MEPS 人口／利用率分群統計 | 最高 0.235160 | [Cell](results/meps_cell_results/summary.csv)；最佳是直接人口特徵，`Cell_All` 0.235131 |
| MEPS 年度 ER/IP 風險直接加入 | 最高 0.234983 | [Utilization](results/meps_utilization_results/summary.csv) `MEPS_Both_2` |
| MEPS 預訓練／多任務 MLP | 0.222629／0.223721 | [多任務](results/meps_multitask_results/summary.csv)，該 MLP 基線 0.222066；不能和 XGBoost 絕對值當同模型比較 |

`Direct8_MEPS_Both8` 在 seed 777／2024／3407 的五折平均 AP 分別為 **0.235585／0.233741／0.234106**；各自 Clinical 為 **0.234919／0.232897／0.233321**。[三組輸出](results/meps_bridge8_results/summary.csv)｜[2024](results/meps_bridge8_seed2024/summary.csv)｜[3407](results/meps_bridge8_seed3407/summary.csv)。這些跨 seed 數值是穩健性檢查，不能用選最好的 seed 來宣稱 SOTA。

其他已執行的 CDC 檢查：較早的 [CDC proxy XGBoost 固定 holdout](results/cdc_proxy_xgb_results/experiment_results.csv)中，Baseline test AP 0.224499、CDC-All 0.226693；這套特徵與分割不同於上表，後續五折結果優先採用上表。[CDC domain-shift 診斷](results/cdc_domain_shift_analysis/domain_results.csv)的病人層來源辨識 AUC 在 seed 777 的完整特徵下為 0.909635，顯示 CDC 與醫院族群差異很大，但它**不是**再住院 PR-AUC。[CDC 編碼器／類別不平衡對照](results/cdc_encoder_control_results/summary_mean_std.csv)中，Bridge 0.232240、CDC-pretrained 0.232009、隨機編碼器 0.232117，SMOTENC 版本約 0.221–0.223；這些分數使用該腳本自己的基線。更早的 [CDC 五折](results/cdc_5fold_results/summary_mean_std.csv)與 [CDC＋不平衡 20 組](results/cdc_ih_5fold_results/summary_mean_std.csv)保留作歷史紀錄；同名方法以較新的臨床／診斷感知結果為主。

### 較保守的臨床特徵流程：MEPS 遷移學習

這組腳本的 Clinical XGBoost 多為 **0.209917**，與上節的 0.234919 是不同特徵流程。年度 MEPS 目標是次年 ER、住院與花費；2008–2012 年有 4,458 個連續年度配對。2008–2016／2008–2024 年增至 9,397／21,176 配對，但樣本變多沒有單調提升 UCI AP。

| 方法 | 五折平均 AP | 同腳本對照與來源 |
|---|---:|---|
| MEPS embedding＋XGBoost | 0.209831 | [五種控制](results/meps_embedding_xgb_results/summary.csv)：Clinical 0.209917、random 0.209019、pretrained 0.209534、shuffled 0.208424 |
| 巢狀調整多任務 embedding | 0.208415 | [巢狀調參](results/meps_embedding_tuned_results/summary.csv)：該腳本 Clinical 0.209449 |
| 年度縱向 MLP 預訓練／MTL | 0.201978／0.201502 | [縱向對齊](results/meps_longitudinal_aligned_results/summary.csv)：MLP `uci_only` 0.200808；真／shuffled 配對見 bootstrap |
| 早期直接共享／預訓練／縱向 MTL MLP | 0.201459／0.201408／0.201473 | [早期縱向](results/meps_longitudinal_results/summary.csv)；後來以上列對齊實作為主 |
| 縱向 MTL 風險直接加 XGBoost | 0.209080 | [風險與 embedding](results/meps_longitudinal_xgb_results/summary.csv)：Clinical 0.209917 |
| 僅加入縱向風險的 XGBoost | 0.209375 | [風險控制](results/meps_longitudinal_risk_xgb_results/summary.csv)：Clinical 0.209917；後續以融合實作為主 |
| 年度縱向 MTL＋XGBoost 驗證集融合 | **0.210155** | [2008–2012 主結果](results/meps_longitudinal_blend_results/summary.csv)：Clinical 0.209917；相對 Clinical 的 pooled delta +0.000275，95% CI −0.000284 至 0.000896 |
| 上項的 shuffled-label 控制 | 0.209519 | 真實融合減 shuffled 融合 pooled delta +0.001443，95% CI +0.000334 至 +0.002628；[matched bootstrap](results/meps_longitudinal_blend_results/external_signal_bootstrap.csv) |
| 保守搜尋融合權重 | 0.209894 | [巢狀權重](results/meps_longitudinal_tuned_blend_results/summary.csv) |
| 縱向融合，擴至 2008–2016／2008–2024 | 0.210137／0.210002 | [2016](results/meps_longitudinal_blend_2008_2016_results/summary.csv)｜[2024](results/meps_longitudinal_blend_2008_2024_results/summary.csv)；未改善 2008–2012 主結果 |
| 跨三 seed 平均縱向 OOF 預測 | 0.210191 | [ensemble](results/meps_longitudinal_blend_ensemble_results/summary.csv)：Clinical 平均預測 0.210040；差距的 CI 跨零 |
| 精確 30 天 MEPS 事件預訓練 | 0.208973 | [事件遷移](results/meps_event_transfer_results/summary.csv)：Clinical 0.209917 |
| MEPS 事件風險直接融合 | 0.209904 | [事件融合](results/meps_event_blend_results/summary.csv)：Clinical 0.209917；單獨事件風險 0.111573／AUROC 0.494876，跨資料域失配 |
| 年度＋事件多來源 MTL | 0.209600 | [多來源融合](results/meps_multisource_blend_results/summary.csv)：Clinical 0.209917；外部事件標籤對 shuffled 有訊號，但未提高最終 AP |

### HGB、XGBoost 集成與 CCS 群組

HGB 使用 46 個原始出院欄位；CCS 方法另加三個診斷分組。這裡使用相同 seed 777 五折。`CCS-only` 的提升主要來自診斷分組表示，**不是 MEPS 外部資料的增益**。

| 方法 | 五折平均 AP | 相同流程的比較與來源 |
|---|---:|---|
| 基本 Clinical HGB／MEPS HGB | 0.235741／0.235495 | [HGB](results/meps_hgb_results/summary.csv)；MEPS 風險沒有改善 |
| 調參後 46 欄 HGB | 0.237247 | [nested HGB](results/meps_nested_hgb_blend_seed777/fold_mean_summary.csv) |
| MEPS XGBoost 調參 | 0.234601 | 同上；低於 HGB |
| HGB＋MEPS XGBoost 固定 50/50 融合 | 0.237485 | 同上；相對 HGB +0.000238，逐折病人 bootstrap CI −0.001029 至 0.001544 |
| 驗證集選權重的 nested 融合 | 0.237015 | 同上；比固定 50/50 低 0.000470，CI −0.000848 至 −0.000085 |
| **CCS-only HGB** | **0.239179** | [CCS 分組](results/meps_ccs_group_hgb_seed777/summary.csv)；目前同組最高 |
| MEPS 次年利用率＋CCS 群組 HGB | 0.238943 | 同上；對 CCS-only −0.000236，CI 跨零 |
| **MEPS 真實 30 天住院事件＋CCS 群組 HGB** | **0.239077** | [事件 CCS](results/meps_event_ccs_group_hgb_seed777/summary.csv)；對 CCS-only −0.000102，CI 跨零；目前同組最高外部病人資料方法 |
| MEPS event-CCS 模糊前綴／階層群組 | 0.237349 | [soft prefix](results/meps_event_ccs_soft_seed777/summary.csv)；shuffled 對照 0.234799；真實訊號存在，但低於簡單 CCS 群組 |
| MEPS 風險特徵擴充 HGB | 0.236877 | [augmented HGB](results/meps_augmented_hgb_seed777/summary.csv)；對 46 欄 HGB −0.000370，CI 跨零 |
| MEPS domain-weighted residual 校正 | 0.237102 | [加權 residual](results/meps_harmonized_weighted_residual_seed777/summary.csv)；對 46 欄 HGB −0.000145，CI 跨零 |
| 未加權的早期 residual 校正 | 0.237245 | [早期 residual](results/meps_residual_correction_seed777/summary.csv)；近似 HGB 0.237247，後續以上列加權版本為主 |
| 醫院群組率先驗／MEPS 群組率先驗 | 0.237010／0.235659 | [群組先驗](results/meps_group_prior_hgb_seed777/summary.csv)；MEPS 版對 CCS-only −0.003520，CI −0.005493 至 −0.001545 |
| 固定 HGB／MEPS 固定排名校正／驗證集門控校正 | 0.238865／0.238504／0.238734 | [排名校正](results/meps_event_rank_correction_seed777/summary.csv)；五折中四折門控選 0，未改善 |
| 擴充 MEPS 2008–2024／NHANES／兩者同加 | 0.238625／0.237414／0.238111 | [真實外部資料比較](results/real_external_sources_hgb_seed777/summary.csv)；均低於同腳本 CCS-only 0.239179；NHANES 下降 CI −0.003545 至 −0.000067 |
| MEPS＋NHANES 分層群組統計／外部多任務編碼器 | 0.237757／0.238303 | [雙來源五折結果](results/meps_nhanes_dual_external_seed777/summary.csv)；單來源消融、六指標及與 CatBoost 原始 ICD-9 融合見[方法與分析](docs/dual_external_meps_nhanes.md)。最佳雙來源融合 0.246877，仍低於 CCS-only＋CatBoost 0.247374；配對 CI 跨零 |
| MEPS 真實 30 天共同訓練／群組專家 CatBoost | 0.219264／0.246291 | [五折結果](results/meps_30d_joint_transfer_seed777/summary.csv)；共同訓練比同架構 UCI-only 神經網路 +0.002644，CI +0.000035 至 +0.005273；群組專家比原始 CatBoost −0.000215，CI 跨零；兩者皆未超過最佳 CCS-only HGB＋CatBoost。[方法、六指標及 SHAP](docs/meps_30d_joint_transfer.md) |
| TabPFN 3.5：CCS-only／CCS＋MEPS | 0.212706／0.210531 | [完整五折與六指標](docs/tabpfn35_results.md)；每折 5,000 筆 context。MEPS 對 CCS-only −0.002175，95% CI −0.003569 至 −0.000832；同 context 的固定 HGB 約 0.191，完整訓練折 HGB／CatBoost 約 0.239／0.247 |
| 三折內層 CV：CCS-only／CCS＋MEPS | 0.237235／0.237369 | [舊版執行](results/meps_ccs_hgb_inner_cv_seed777/summary.csv)；**舊程式跑滿 250–400 輪，與先前的早停方式不一致**，已修正程式但尚無修正版結果，不據此判斷三折較差 |

跨 seed 的 HGB／XGBoost OOF 平均是**另一種統計口徑**：[三 seed pooled 結果](results/meps_hgb_xgb_ensemble/summary.csv)為 HGB 0.238323、無外部資料集成 0.237798、有 MEPS 集成 0.238312。配對病人 bootstrap 顯示有 MEPS 對無外部資料 +0.000514（95% CI +0.000108 至 +0.000924），但有 MEPS 與 HGB 相差 −0.000011（CI −0.001417 至 0.001413）。

## 尚未有完整結果

- [修正版三折內層 CV 執行指南](docs/pending_experiments.md)已補齊最新五折稽核與正式指令；仍無可核對的修正版完整五折分數。TabPFN 3.5 已完成，結果見[專文](docs/tabpfn35_results.md)。
- LR 可由 [共用 baseline runner](train_baselines.py)重新執行；目前 README 所列 LR 數字來自使用者提供的早期紀錄，未找到對應 metrics／log／預測檔。

## 重現與檔案位置

公開倉庫保留程式、執行 log 與小型指標摘要。原始／處理後資料、逐筆預測和模型權重不納入 Git；取得資料後放到下列目錄，依 `scripts/experiments/` 的準備腳本重建處理資料。README 連結的結果摘要會隨程式一併上傳。

本專題自行撰寫的程式以 [MIT License](LICENSE) 授權。UCI、MEPS、CDC、NHANES、AHRQ 等第三方資料與對照表不受此授權涵蓋，使用時須遵守各來源的條款。

| 路徑 | 用途 |
|---|---|
| `data/raw/` | UCI、MEPS、CDC、NHANES 與 AHRQ 原始資料 |
| `data/processed/` | 已整理的年度 MEPS、MEPS 事件、NHANES 與 UCI 分割資料 |
| `results/` | 每折指標、OOF 預測、bootstrap、模型與設定；本 README 的數字來源 |
| `logs/` | 早期執行 log；`rf.log`、`xgboost.log` 可核對舊 holdout 結果 |
| `src/diabetes_project/` | 共用資料處理、分割、特徵與模型程式 |
| `scripts/experiments/` | CDC、MEPS、CCS、HGB、TabPFN 專題實驗 |
| `archive/legacy_scripts/` | 舊版單一方法腳本，僅保留追溯用途 |

從本目錄重建 patient-disjoint 五折資料：

```bash
python preprocess.py
```

輸出於 `data/processed/uci_5fold/`，含 outer train／test、inner train／validation 與 `fold_assignments.csv`；預處理器只在相應訓練側擬合。`--overwrite` 只在確定要重建時使用。早期固定 holdout 的結果來自另一組已存的 train／validation／test，不是上述五折的單一折。

要在已建立的某一折執行 LR、RF、XGBoost 共用 baseline：

```bash
python train_baselines.py --fold 1 --models lr rf xgb
```

要在**近期論文比較使用的 99,343 筆、69,990 位病人五折分割**上，一次重跑早期 LR（標準與 balanced）、RF（標準與 balanced）、XGBoost（標準、class weight、四種 SMOTENC 比例），使用：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/experiments/run_classic_5fold.py \
  --reference-oof results/meps_ccs_group_hgb_seed777/oof_all.csv \
  --outdir results/classic_5fold_recheck_seed777 \
  --max-folds 5 --device cuda --seed 777
```

這支腳本核對近期 OOF 的 encounter、病人、標籤與 fold，並產生十種方法的 `summary.csv`、`fold_metrics.csv` 和 `oof_long.csv`。現有 `data/processed/uci_5fold/` 是較早的 **99,340 筆**版本，不能直接和近期 99,343 筆結果混為同一分割比較。LR、RF 使用 CPU，XGBoost 使用指定的 GPU。詳見 [實驗指令](scripts/experiments/README.md)。

原始實驗與完整數字請以各結果目錄的 `config.json`、`summary.csv`／`summary_mean_std.csv`、`fold_results.csv`、`oof_all.csv` 和 bootstrap CSV 為準。對外報告時請同時寫明**資料定義、分割方式、平均或 pooled AP、是否使用外部病人資料**。
