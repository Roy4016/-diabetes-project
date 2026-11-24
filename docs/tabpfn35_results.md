# TabPFN 3.5：完整五折結果

來源：[五折摘要](../results/tabpfn35_meps_seed777/summary.csv)、[逐折結果](../results/tabpfn35_meps_seed777/fold_results.csv)、[病人配對 bootstrap](../results/tabpfn35_meps_seed777/fold_stratified_patient_bootstrap.csv)、[設定](../results/tabpfn35_meps_seed777/config.json)。

## 實驗設定

- UCI 醫院資料 99,343 筆就醫、69,990 位病人；seed 777、病人互斥五折。與既有 HGB／CatBoost OOF 的 encounter、病人、標籤及折別完全一致；每個模型完成 99,343 筆 OOF 預測，沒有缺值或重複 encounter。
- TabPFN 套件 10.2.0、模型 3.5、本機 CUDA；每折從外層訓練資料抽 5,000 筆 context，`n_estimators=2`。比較 CCS-only 與 CCS＋真實 MEPS 住院事件 CCS 群組特徵。
- `context_hgb_*` 使用相同 5,000 筆 context，但採固定 HGB 設定。`hgb_tuned`、`ccs_only_hgb` 及 CatBoost 對照使用完整外層訓練資料，不能視為相同訓練量比較。
- 主要指標 PR-AUC (Average Precision, AP) 為 `average_precision_score` 的五折平均。Precision、Recall、F1 用各折測試集最高風險 10% 警示量；Brier、AUROC、AP 不依賴此警示門檻。`Pooled AP` 為合併 OOF 後重算，與五折平均不同。

## 完整指標

| 方法 | AP 平均 | AP SD | AUROC | Precision | Recall | F1 | Brier | Pooled AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HGB tuned，完整訓練折 | 0.237247 | 0.002473 | 0.676933 | 0.279557 | 0.245358 | 0.261343 | 0.095573 | 0.235987 |
| CCS-only HGB，完整訓練折 | 0.239179 | 0.004208 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 | 0.237823 |
| CatBoost raw，完整訓練折 | 0.246506 | 0.004438 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 | 0.245430 |
| CCS-only HGB＋CatBoost 50/50 | **0.247374** | 0.004744 | **0.683446** | **0.288620** | **0.253313** | **0.269817** | **0.095068** | **0.245932** |
| Context HGB，CCS-only | 0.190944 | 0.004110 | 0.638954 | 0.235650 | 0.206823 | 0.220297 | 0.098440 | 0.190561 |
| Context HGB，CCS＋MEPS | 0.190692 | 0.002977 | 0.640043 | 0.236153 | 0.207266 | 0.220769 | 0.098449 | 0.190313 |
| TabPFN 3.5，CCS-only | 0.212706 | 0.005151 | 0.660754 | 0.256596 | 0.225206 | 0.239879 | 0.096989 | 0.209293 |
| TabPFN 3.5，CCS＋MEPS | 0.210531 | 0.004557 | 0.660190 | 0.254179 | 0.223087 | 0.237620 | 0.097072 | 0.207745 |
| TabPFN CCS-only＋CatBoost 50/50 | 0.237858 | 0.005091 | 0.677665 | 0.279053 | 0.244917 | 0.260873 | 0.095630 | 0.236310 |
| TabPFN CCS＋MEPS＋CatBoost 50/50 | 0.237373 | 0.004666 | 0.677802 | 0.280967 | 0.246596 | 0.262662 | 0.095658 | 0.235948 |

## 配對病人 bootstrap

以下為五折內依病人重抽 1,000 次的平均 AP 差與 95% 區間：

| 比較 | AP 差 | 95% CI | 正差比例 |
|---|---:|---:|---:|
| TabPFN CCS＋MEPS − TabPFN CCS-only | **−0.002175** | [−0.003569, −0.000832] | 0.000 |
| TabPFN CCS-only − 同 context HGB | +0.021763 | [+0.017637, +0.025656] | 1.000 |
| TabPFN CCS＋MEPS − 同 context HGB | +0.019839 | [+0.015551, +0.023740] | 1.000 |
| TabPFN CCS＋MEPS − 完整訓練折 CCS-only HGB | −0.028648 | [−0.033603, −0.024368] | 0.000 |
| TabPFN CCS＋MEPS＋CatBoost − CCS-only HGB＋CatBoost | −0.010001 | [−0.012272, −0.007751] | 0.000 |
| TabPFN CCS＋MEPS＋CatBoost − TabPFN CCS-only＋CatBoost | −0.000485 | [−0.001016, +0.000180] | 0.086 |

## 解讀

TabPFN 在固定 5,000 筆 context 下，明顯高於同資料量的**固定參數** HGB；這不能推論它優於使用完整約 79,000 筆外層訓練資料的 HGB 或 CatBoost。MEPS 群組特徵使 TabPFN 的 AP 下降約 0.0022，區間全為負，因此這個接法在本次設定沒有改善。與 CatBoost 50/50 融合也沒有超過 CCS-only HGB＋CatBoost 的 0.247374。所有 TabPFN 結果均使用相同的病人互斥五折，但 TabPFN 的 5,000 筆 context 和 HGB／CatBoost 的完整訓練量不同；報告時須同時註明。
