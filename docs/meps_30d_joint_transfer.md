# MEPS 30 天再住院：群組專家與共同訓練

這個實驗使用真實 MEPS 2008–2012 住院事件，以**出院後 30 天內再次住院**作外部標籤。主資料仍是 UCI 99,343 次就醫、69,990 位病人的固定 seed 777、病人互斥五折；與既有 CCS-only HGB 和原始 ICD-9 CatBoost 的逐筆 OOF 對齊。MEPS 有 2,688 筆標籤可確定的住院事件、1,657 位病人、359 筆 30 天再住院。程式會在執行時從資料重新計數。

## 兩種接法

1. **共同訓練**：UCI 與 MEPS 以年齡、性別、種族、住院天數及常見 CCS 診斷群組共用一個小型神經網路編碼器。UCI 另有 46 個原始出院欄位的專屬分支；MEPS 有事件專屬分支。兩個來源有各自的 30 天標籤預測頭。UCI loss 每批更新；每隔四批加入一次 MEPS loss。`uci_only_nn` 使用完全相同架構、訓練輪數與隨機種子，只停用 MEPS loss，作為外部資料增益的消融。
2. **群組專家**：僅用 MEPS 事件的上述共同欄位訓練 CatBoost 30 天再住院風險模型。對每筆 UCI 就醫計算其 MEPS 風險與所屬 CCS 群組的來源支持量，再和 46 個出院欄位一起交給 UCI CatBoost 訓練。用既有 `catboost_raw` OOF 作無外部資料對照；另將兩種 CatBoost 與固定的 CCS-only HGB 以 50/50 融合。

共同欄位是出院時可得資訊。MEPS 的下一次住院日期只用於**標籤**；不作模型輸入。MEPS 住院事件與 UCI 個案沒有共同病人。MEPS 專家模型的內部診斷採 MEPS 病人互斥 80/20 切分；最後再以全部 MEPS 事件擬合，因為它不使用任何 UCI 標籤。

## 執行

```bash
cd /media/synology/G/314554044/diabetes_project
python -m pip install -e '.[comparison,dual_external]'

python scripts/experiments/run_meps_30d_joint_transfer.py \
  --outdir results/meps_30d_joint_transfer_seed777 \
  --max-folds 5 \
  --epochs 12 \
  --bootstrap 1000
```

單折檢查可加 `--max-folds 1 --epochs 1 --bootstrap 0 --outdir results/meps_30d_joint_smoke`；單折結果不可與完整五折分數比較。預設在 CPU 使用四條執行緒，避免占用 GPU。

輸出 `summary.csv`、`fold_metrics.csv`、`paired_ap_bootstrap.csv`、`training.csv`、`config.json` 和逐筆 OOF。指標包含平均 **PR-AUC (Average Precision, AP)、AUROC、Precision、Recall、F1、Brier**；另存 pooled AP。Precision、Recall、F1 使用每折固定最高風險 10% 警示量。Bootstrap 以每折病人為重抽單位，報告五折平均 AP 差及 95% 區間。`shap_fold.csv`／`shap_global.csv` 提供加入外部特徵的 CatBoost TreeSHAP；預設每折抽 500 筆、單位為 log-odds。可用 `--shap-sample 0` 略過 SHAP。

主要判讀：`joint_meps_30d` 對 `uci_only_nn` 檢驗真正共同訓練的增益；`group_teacher_catboost` 對 `catboost_raw` 檢驗 MEPS 群組專家是否改善 CatBoost；再檢查兩種固定 50/50 融合是否超過 `ccs_only_hgb_catboost50`。相對文獻的 0.237 只能作數值參考，因文獻採不同 holdout 切分。

## 單折執行檢查

已用正式的 12 epoch、400 棵 MEPS 專家樹和 750 棵 UCI CatBoost 樹跑完**第 1 折**，確認主要訓練與評估流程。該折 AP 為 UCI-only 神經網路 0.218649、UCI＋MEPS 共同訓練 0.221993；原始 CatBoost 0.250130、加入 MEPS 專家特徵的 CatBoost 0.249454；CCS-only HGB＋原始 CatBoost 0.249205。MEPS 專家在自身病人互斥驗證集的 AP 只有 0.148036、AUROC 0.491444，顯示這條來源模型訊號偏弱。**以上都只是第 1 折，不能作五折結論。**另用小樹數檢查了 SHAP 檔案能正常產生；小樹數分數不作比較。

## 完整五折結果（2026-11-22）

使用預設設定完成全部五折。下列為每折指標的平均；AP 是 `average_precision_score`，Precision／Recall／F1 採每折最高風險 10% 的固定警示量。`Pooled AP` 是另一種計算口徑，不能與五折平均 AP 混排。[原始摘要](../results/meps_30d_joint_transfer_seed777/summary.csv)｜[逐折指標](../results/meps_30d_joint_transfer_seed777/fold_metrics.csv)

| 模型 | AP 平均 | AP 標準差 | AUROC | Precision | Recall | F1 | Brier | Pooled AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CCS-only HGB | 0.239179 | 0.004208 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 | 0.237823 |
| 原始 ICD CatBoost | 0.246506 | 0.004438 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 | 0.245430 |
| **CCS-only HGB＋CatBoost** | **0.247374** | 0.004744 | **0.683446** | **0.288620** | **0.253313** | **0.269817** | **0.095068** | **0.245932** |
| UCI-only 神經網路 | 0.216620 | 0.005942 | 0.664481 | 0.263243 | 0.231040 | 0.246093 | 0.097245 | 0.214873 |
| **UCI＋MEPS 共同訓練** | **0.219264** | 0.004465 | 0.665757 | 0.263746 | 0.231482 | 0.246563 | 0.097049 | 0.217643 |
| UCI-only 神經網路＋CatBoost | 0.238045 | 0.005711 | 0.680662 | 0.280262 | 0.245978 | 0.262003 | 0.095510 | 0.236593 |
| UCI＋MEPS 共同訓練＋CatBoost | 0.239537 | 0.004790 | 0.680805 | 0.283686 | 0.248983 | 0.265204 | 0.095406 | 0.238232 |
| MEPS 群組專家＋CatBoost | 0.246291 | 0.005602 | 0.682596 | 0.285498 | 0.250572 | 0.266898 | 0.095091 | 0.245139 |
| MEPS 群組專家 CatBoost＋CCS-only HGB | 0.247038 | 0.005608 | 0.683300 | 0.288218 | 0.252959 | 0.269440 | 0.095077 | 0.245591 |

逐折病人配對 bootstrap（1,000 次）結果：[完整 CSV](../results/meps_30d_joint_transfer_seed777/paired_ap_bootstrap.csv)

| 比較（前者減後者） | 平均折 AP 差 | 95% CI |
|---|---:|---:|
| 共同訓練 − UCI-only 神經網路 | +0.002644 | [+0.000035, +0.005273] |
| 共同訓練＋CatBoost − UCI-only 神經網路＋CatBoost | +0.001492 | [+0.000052, +0.003146] |
| MEPS 群組專家 CatBoost − 原始 CatBoost | −0.000215 | [−0.001495, +0.001029] |
| MEPS 群組專家 CatBoost＋HGB − CCS-only HGB＋CatBoost | −0.000336 | [−0.001024, +0.000405] |
| 共同訓練＋CatBoost − CCS-only HGB＋CatBoost | −0.007836 | [−0.010213, −0.005342] |

**判讀**：真實 MEPS 30 天資料對神經網路自身有小幅正向訊號；前兩個區間下界很接近零，這是單次設定與多組探索性比較，不宜稱為穩健的最終進步。MEPS 群組專家沒有改善 CatBoost；兩種外部資料版本都沒有超過同切分最高的 CCS-only HGB＋CatBoost。來源專家在 MEPS 自身病人互斥驗證的 AP 0.148036、AUROC 0.491444，也支持其可遷移訊號很弱的判讀。

加入外部特徵的 CatBoost TreeSHAP 在每折抽 500 筆、合計 2,500 筆；`meps_30d_teacher_risk` 排第 **8**（平均絕對 SHAP 0.032326 log-odds），`meps_ccs_log_support` 排第 **15**（0.019019）。主要特徵仍為 `number_inpatient`（0.230564）和 `discharge_disposition_id`（0.146597）。SHAP 顯示模型使用了外部特徵，**不代表它帶來性能增益或因果影響**。[完整 SHAP](../results/meps_30d_joint_transfer_seed777/shap_global.csv)
