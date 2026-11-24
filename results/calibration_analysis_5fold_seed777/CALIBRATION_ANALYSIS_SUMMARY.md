# 五折校準與預測機率可靠度分析

## 設定與 OOF 核對

UCI 共 **99,343** 次就醫、**69,990** 位病人；正例 **11,314** 筆（11.3888%）。使用 seed 777 的病人互斥五折 OOF 原始機率，沒有重新訓練或校準模型。Encounter ID、Patient ID、Fold、Label 均完整；原始 CatBoost 與 CCS HGB＋Raw CatBoost機率已逐筆和論文比較 OOF 核對。

10 箱 reliability diagram 使用 `sklearn.calibration.calibration_curve(strategy='quantile')`，每個模型按自己的機率分位數分箱。定量 ECE 和配對 bootstrap 使用**各折原融合模型**的 10 個分位箱邊界，並將同一組邊界套到三個模型。因此圖中的分箱與 ECE 分箱不同；`calibration_bin_statistics.csv` 用 `Binning` 明列兩者。ECE = Σ_b (n_b/N)|observed_b − predicted_b|；空箱不補值。

## 三模型比較

| 模型 | 五折平均 Brier ↓ | 五折 Brier SD | Pooled Brier ↓ | 五折平均 ECE ↓ | 五折 ECE SD | Pooled ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Raw ICD-9 CatBoost | 0.095084 | 0.000200 | 0.095084 | 0.006964 | 0.000540 | 0.004801 |
| CCS HGB + Raw CatBoost (50/50) | 0.095068 | 0.000240 | 0.095068 | 0.007441 | 0.001883 | 0.004851 |
| CDC + MEPS + CMS CatBoost + CCS HGB (50/50) | 0.095073 | 0.000294 | 0.095073 | 0.007280 | 0.001295 | 0.005360 |

五折平均與 pooled OOF 是不同的估計方式，已分開報告。

## 配對病人 bootstrap

比較方向固定為 **外部資料融合 − 原融合模型**；Brier、ECE 差值小於零表示較低。每折以病人為抽樣單位，兩模型使用相同抽樣；先算各折差值再平均。ECE 的箱界固定為原始完整 OOF 的各折分位數，bootstrap 只重新計算箱內人數、正例率與平均機率，**不重新估計箱界**。

| 指標 | 平均五折差值 | 95% CI |
|---|---:|---:|
| Brier | +0.000006 | [-0.000036, +0.000053] |
| ECE | -0.000161 | [-0.001567, +0.001684] |

CI 跨零不能證明兩模型等效。Brier 同時受排序、機率誤差及結果不確定性影響；ECE 依分箱策略而變，不能單獨證明機率可靠。

## 曲線解讀與限制

- **Raw ICD-9 CatBoost**：整體平均預測 0.1127、觀察率 0.1139，屬於**整體偏低估**；10 箱最大絕對偏差 0.0091。仍須看不同風險箱是否交叉或局部偏離。
- **CCS HGB + Raw CatBoost (50/50)**：整體平均預測 0.1130、觀察率 0.1139，屬於**整體偏低估**；10 箱最大絕對偏差 0.0141。仍須看不同風險箱是否交叉或局部偏離。
- **CDC + MEPS + CMS CatBoost + CCS HGB (50/50)**：整體平均預測 0.1129、觀察率 0.1139，屬於**整體偏低估**；10 箱最大絕對偏差 0.0193。仍須看不同風險箱是否交叉或局部偏離。

圖：`calibration_curve_pooled.png`、`calibration_curve_fivefold.png`。分位箱的模型各自邊界讓每箱樣本數接近，但曲線點不能視為共同風險區間。定量比較使用共同箱界；其他模型在某些箱可能很少或沒有樣本。本分析未檢驗族群或醫院間的校準，也未做外部前瞻驗證。

## English slide-ready conclusion

- Across five patient-disjoint folds, the original ensemble had a mean Brier score of 0.095068 and common-bin ECE of 0.007441.
- The CDC + MEPS + CMS ensemble had a mean Brier score of 0.095073 and common-bin ECE of 0.007280.
- Paired patient bootstrap quantifies the differences; these plots assess reliability of existing probabilities, without recalibration.

## 重現指令

```bash
cd /media/synology/G/314554044/diabetes_project
python scripts/experiments/analyze_calibration_5fold.py \
  --oof results/cms_meps_cdc_catboost_seed777/oof_all.csv.gz \
  --paper-oof results/paper_fair_comparison_seed777/oof_all.csv.gz \
  --outdir results/calibration_analysis_5fold_seed777 \
  --bootstrap 2000 --seed 777
```
