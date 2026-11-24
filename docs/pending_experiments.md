# TabPFN 3.5 已完成與修正版內層 CV 待完成

兩項實驗均使用既有 **99,343 筆就醫、69,990 位病人、seed 777 的 patient-disjoint 五折**。執行前會檢查 encounter、patient、label 和每筆所屬折是否與既有 OOF 相同。外部資料使用真實 MEPS 2008–2012 住院事件及事件連結的 CCS 診斷，產生 30 天再住院／急診群組率；AHRQ CCS 對照表本身不是外部病人資料。

## 1. TabPFN 3.5

**已完成** 5 折正式實驗，結果與六項指標見[TabPFN 3.5 分析](tabpfn35_results.md)。以下指令保留供重現。

程式：[run_tabpfn35_meps.py](../scripts/experiments/run_tabpfn35_meps.py)。比較 CCS-only 與 CCS＋MEPS 事件群組特徵，各自用相同 context 筆數訓練 HGB 作公平的資料量對照。另列完整外層訓練資料的既有 HGB、原始 ICD-9 CatBoost、CCS-only HGB＋CatBoost，以及 TabPFN 與 CatBoost 固定 50/50 融合；完整 HGB 與 context HGB 不應混稱同一訓練量。

程式已補上 **AP／PR-AUC、AUROC、Precision、Recall、F1、Brier、pooled AP**，Precision／Recall／F1 採每折預先固定的最高風險 10% 警示量；完成五折後做折內病人配對 bootstrap。每折保存 OOF，若中斷可用相同參數加 `--resume`，會核對設定與既有預測的 ID、病人、標籤、折別。

先跑只讀資料稽核，無須安裝 TabPFN 或占用 GPU：

```bash
cd /media/synology/G/314554044/diabetes_project
python scripts/experiments/run_tabpfn35_meps.py \
  --audit-only --outdir results/tabpfn35_split_audit_seed777
```

安裝與單折 pilot：

```bash
python -m pip install -e '.[tabpfn]'
CUDA_VISIBLE_DEVICES=5 python scripts/experiments/run_tabpfn35_meps.py \
  --device cuda --context-rows 5000 --n-estimators 2 \
  --max-folds 1 --test-limit 500 --bootstrap 0 \
  --outdir results/tabpfn35_meps_pilot_seed777
```

選擇有空間的 GPU 編號；如果 GPU 忙碌，可改 `--device cpu` 做小型 pilot。**pilot 只評估第一折的 500 筆測試列，不是正式五折結果。**

正式五折：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/experiments/run_tabpfn35_meps.py \
  --device cuda --context-rows 5000 --n-estimators 2 \
  --max-folds 5 --test-limit 0 --bootstrap 1000 \
  --outdir results/tabpfn35_meps_seed777
```

若中斷，重複相同指令並加 `--resume`。想先試較快的 3.5 Fast，請用**另一個 outdir** 並加 `--model-version 3.5-fast`；兩種模型的結果需分開報告。TabPFN 首次執行需要下載權重並接受 Prior Labs 權重授權；沒有圖形介面的環境可依[官方文件](https://github.com/PriorLabs/TabPFN#installation--setup)於其網站接受後設定 `TABPFN_TOKEN`，請勿把 token 放進 Git。官方建議 GPU；CPU 的 3.5 context 預設上限 5,000 筆，程式也會限制。TabPFN 的 `fit_with_cache` 可重用 context 表示，降低分塊預測的重複計算。[官方使用說明](https://github.com/PriorLabs/TabPFN#usage-tips)

正式執行環境已安裝 TabPFN 10.2.0，使用 3.5 權重與 CUDA 完成五折。若在其他環境重現，仍需自行接受相應模型權重授權。

## 2. 修正版三折內層 CV

程式：[run_meps_ccs_hgb_inner_cv.py](../scripts/experiments/run_meps_ccs_hgb_inner_cv.py)。原始結果的訓練輪數與後續早停口徑不一致；現版本在每個外層訓練折中，用病人互斥的三個內層折選 HGB 參數及輪數，再擬合外層訓練資料。比較 CCS-only 與 CCS＋MEPS；已補上六項指標、pooled AP、折內病人 bootstrap 及事前分割稽核。它需要多次 HGB 擬合，耗時會比單一五折實驗長。

```bash
python scripts/experiments/run_meps_ccs_hgb_inner_cv.py \
  --audit-only --outdir results/meps_ccs_hgb_inner_cv_audit_seed777

python scripts/experiments/run_meps_ccs_hgb_inner_cv.py \
  --reference-oof results/meps_event_ccs_group_hgb_seed777/oof_all.csv \
  --outdir results/meps_ccs_hgb_inner_cv_corrected_seed777 \
  --folds 5 --inner-folds 3 --bootstrap 1000 --seed 777
```

請將**修正版結果放在新目錄**，保留舊版結果供方法差異追溯。只有完整五折並產出 `summary.csv` 後，才可更新正式結果表；執行過程或 mock 驗證不能當作已完成分數。
