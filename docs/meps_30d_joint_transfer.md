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

輸出 `summary.csv`、`fold_metrics.csv`、`paired_ap_bootstrap.csv`、`training.csv`、`config.json` 和逐筆 OOF。指標包含平均 **PR-AUC/AP、AUROC、Precision、Recall、F1、Brier**；另存 pooled AP。Precision、Recall、F1 使用每折固定最高風險 10% 警示量。Bootstrap 以每折病人為重抽單位，報告五折平均 AP 差及 95% 區間。`shap_fold.csv`／`shap_global.csv` 提供加入外部特徵的 CatBoost TreeSHAP；預設每折抽 500 筆、單位為 log-odds。可用 `--shap-sample 0` 略過 SHAP。

主要判讀：`joint_meps_30d` 對 `uci_only_nn` 檢驗真正共同訓練的增益；`group_teacher_catboost` 對 `catboost_raw` 檢驗 MEPS 群組專家是否改善 CatBoost；再檢查兩種固定 50/50 融合是否超過 `ccs_only_hgb_catboost50`。相對文獻的 0.237 只能作數值參考，因文獻採不同 holdout 切分。

## 單折執行檢查

已用正式的 12 epoch、400 棵 MEPS 專家樹和 750 棵 UCI CatBoost 樹跑完**第 1 折**，確認主要訓練與評估流程。該折 AP 為 UCI-only 神經網路 0.218649、UCI＋MEPS 共同訓練 0.221993；原始 CatBoost 0.250130、加入 MEPS 專家特徵的 CatBoost 0.249454；CCS-only HGB＋原始 CatBoost 0.249205。MEPS 專家在自身病人互斥驗證集的 AP 只有 0.148036、AUROC 0.491444，顯示這條來源模型訊號偏弱。**以上都只是第 1 折，不能作五折結論。**另用小樹數檢查了 SHAP 檔案能正常產生；小樹數分數不作比較。
