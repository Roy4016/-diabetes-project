# HCUPnet 與 CMS：疾病 × 年齡再住院外部訊號可行性

更新：2026-11-24。此文件是下一輪實驗規格，**尚無 HCUPnet／CMS 加入後的五折結果**。主資料仍是 UCI 糖尿病住院就醫，維持既有 seed 777 病人互斥五折，以平均 PR-AUC (Average Precision, AP)為主要判準。

## 資料能提供什麼

| 來源 | 公開資料的單位 | 30 天再住院結果 | 疾病與年齡 | 在本專題的用途 |
|---|---|---|---|---|
| AHRQ HCUPnet Readmissions（NRD 彙總） | 全國疾病群組等彙總表 | 7 天與 30 天再住院率、就醫次數；非逐筆病人標籤 | 可查 CCS／CCSR 等疾病分類；需在 Readmissions 儀表板確認同一匯出表是否同時有年齡分層 | 首選：疾病群組 × 年齡風險先驗；若不能交叉查詢，只做疾病群組先驗 |
| CMS Provider Data Catalog：Unplanned Hospital Visits | 醫院 × 指標 | 指定疾病／手術的 30 天*風險標準化、非計畫性*再住院指標 | 有 AMI、COPD、HF、肺炎等指標；公開表沒有逐筆年齡、診斷或就醫標籤 | 少數疾病的額外先驗或敏感度分析；不當成病人層訓練集 |
| HCUP NRD 原始資料／CMS 受限理賠資料 | 逐筆就醫 | 可建立個別 30 天標籤 | 有較完整疾病、年齡資訊 | 需另取得資料；不屬於目前免費公開的 HCUPnet／CMS 表 |

HCUPnet 的 30 天定義是出院後 30 天內再次住院，可跨同院與不同院，分母為符合追蹤條件的 index stays；HCUPnet 使用彙總表保護紀錄，不能把匯出的比率當作逐筆患者標籤。2023 年起 NRD 的再住院資料限成人，因此跨年前後的年齡比較要分開。[HCUPnet 方法](https://datatools.ahrq.gov/hcupnet/methodology/)、[HCUPnet FAQ](https://hcup-us.ahrq.gov/tech_assist/faq.jsp)、[HCUPnet 工具](https://datatools.ahrq.gov/hcupnet/)。

CMS 公開的 Unplanned Hospital Visits 表有醫院、指標、分母、分數、患者數等欄位，沒有同一患者的年齡與診斷交叉資料。CMS 指標主要面向 Medicare 65 歲以上患者，且使用非計畫性及風險標準化定義，與 UCI 的原始 30 天再住院標籤不同。[CMS 資料集](https://data.cms.gov/provider-data/dataset/632h-zaca)、[CMS 資料字典](https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/hospital/HOSPITAL_Data_Dictionary.pdf)、[CMS 指標說明](https://data.cms.gov/provider-data/topics/hospitals/unplanned-hospital-visits/)。CMS synthetic claims 雖有類似逐筆理賠的格式，卻不是實際病人，不符合本專題「真實 external data」的比較目的。[CMS SynPUF 說明](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files)。

## 先檢查 HCUPnet 匯出是否真的含疾病 × 年齡

### 下載與存放

CMS 的兩張真實彙總表可在有外網的環境從專題根目錄執行：

```bash
python scripts/experiments/download_cms_readmissions.py
```

腳本透過 CMS 官方 API 逐頁下載完整的 hospital 與 national 表，輸出到 `data/raw/cms/unplanned_hospital_visits_{hospital,national}.csv`，並各寫一份含來源網址、抓取時間、列數與 SHA-256 的 `.source.json`。下載不完整時不會留下正式 CSV。`data/raw/` 在 `.gitignore` 中，不會被提交到 GitHub。[CMS API 說明](https://data.cms.gov/provider-data/about)。

HCUPnet 目前須在網站互動查詢後使用 **Download Data** 匯出 Excel；把原始 Excel 放到 `data/raw/hcupnet/`，檔名應含年份、30 天、principal diagnosis、CCS／CCSR 及年齡分層，例如 `hcupnet_nrd_2014_30d_principal_ccs_by_age.xlsx`。此處只保存真實下載檔；不要用合成或示範數據填補。

2026-11-24 已核對 CMS 下載：醫院表 **67,018** 列、全國表 **14** 列，兩者的來源 SHA-256 與 CSV 一致。執行 `python scripts/experiments/prepare_cms_readmission_prior.py` 後，四個可近似映射到 UCI 主要診斷的全國率為：心肌梗塞 14.4%、心衰竭 21.3%、肺炎 17.3%、COPD 20.0%；UCI `diag_1` 覆蓋 **16,413／99,343（16.5%）**。這些是 2023-07 至 2025-06 的 CMS 風險標準化彙總指標，不是 UCI 病人的真實再住院機率；CABG、髖／膝置換因 UCI 缺少對應手術碼而不映射。整理檔在 `data/processed/cms/cms_readmission_ccs_prior.csv`，附來源與覆蓋率稽核。HCUPnet 原始匯出仍未放入 `data/raw/hcupnet/`。

使用 [HCUPnet 的 National Inpatient Readmissions — Diagnoses and Procedures 直連](https://datatools.ahrq.gov/hcupnet?tab=readmissions&dash=48)，選 30 天、疾病分類、年份，嘗試以年齡組分層，按 **Download Data** 匯出 Excel。保留原始檔與查詢條件：年份、CCS／CCSR 版本、診斷是 principal 還是 any-listed、年齡分組、index stay 次數、再住院次數、比率與 suppression 標記。HCUPnet 官方說明可下載 Excel；部分分類與交叉查詢受網站限制。**若同一匯出檔沒有疾病 × 年齡的分母與再住院率，不應把兩個邊際比率相乘假裝成交叉資料。**[HCUPnet 匯出 FAQ](https://datatools.ahrq.gov/hcupnet/faq/)、[2014 NRD 文件](https://hcup-us.ahrq.gov/db/nation/nrd/NRD_Introduction_2014.jsp)。

若 **2014 查詢顯示空白**，先進入 [Readmissions — All Stays](https://datatools.ahrq.gov/hcupnet?tab=readmissions&dash=47) 查 2014 年、30 天、全部年齡，確認基本表有資料；再回診斷儀表板，依序選 2014、principal diagnosis CCS、單一疾病群組、年齡組。每加一項條件即檢查表格是否仍有資料。若基本表也空白，換桌面版 Chrome／Firefox 等瀏覽器並重設篩選；2014 NRD 的存在由官方文件確認，不應將空畫面誤作不存在。若只有交叉後空白，可先匯出同年的疾病總率並記錄「無年齡交叉」，再試 2013 或 2012 年的 ICD-9 CCS。不要用 2016 年以後的 ICD-10 CCSR 數字直接冒充 ICD-9 CCS。

優先使用與 UCI ICD-9-CM 年代較接近的 HCUPnet **ICD-9 CCS** 年份；2016 年起改為 ICD-10-CM 的 CCSR，不能把數字代碼直接視為同一疾病。UCI `diag_1`、`diag_2`、`diag_3` 已有 ICD-9 到 CCS 映射程式，可沿用同一對照表。若使用較新的 CCSR，需另建立 ICD-9 CCS ↔ ICD-10 CCSR 概念級映射，並檢查每個映射的歧義。[HCUPnet 分類方法](https://datatools.ahrq.gov/hcupnet/methodology/)、[本專題 CCS 程式](../scripts/experiments/run_meps_ccs_group_hgb.py)。

## 如何檢驗「併發症比糖尿病本身重要」

這是**預測假說**，不能僅由 SHAP 或相關性推論再住院的因果主因。比較時保留 UCI 的出院前可得資訊，分別做：

1. 原始診斷 `diag_1`、`diag_2`、`diag_3` 與年齡、既往住院、出院去向等臨床變數。
2. 加入 CCS 疾病群組，但沒有外部病人資料的對照。
3. 加入 HCUPnet 的**principal diagnosis CCS × 年齡** 30 天再住院率；另測疾病群組率、年齡率與兩者交叉率各自帶來的差異。對小分母的交叉格用群組／全國率平滑，suppressed 格只用上層率或缺失標記。
4. 加入 CMS 疾病指標先驗；只映射有清楚對應的指標，並保留「無對應」狀態。CMS 無法提供本研究需要的疾病 × 年齡交叉標籤。
5. HCUPnet＋CMS 同時加入，和各自單獨加入比較；同時做 `diag_2`／`diag_3` 消融，觀察共病資訊及外部先驗是否提供增量。

HCUPnet principal diagnosis 的率對應 `diag_1` 最直接。`diag_2`／`diag_3` 是次要診斷，不能直接將 principal-diagnosis 的率稱為「該共病造成的再住院率」；可將其疾病群組當成模型互動特徵，並另做有／無次要診斷的消融。UCI 的診斷欄位未必完整記錄全部併發症，結果應稱「已記錄診斷／共病訊號的預測貢獻」。

## 公平比較與成功條件

- 固定本專題最新的 99,343 筆、69,990 人、seed 777 病人互斥五折，不碰測試折選特徵、參數或門檻。和同一折的 `ccs_only_hgb`、`ccs_meps_group_hgb`、`catboost_raw`、`ccs_only_catboost_raw_50` 以及最佳含 MEPS 方法配對比較。[目前完整五折報告](../RESULTS_ANALYSIS.md)。
- 主指標是五折平均 AP；同時列 AUROC、Precision、Recall、F1、Brier。Precision／Recall／F1 沿用報告的每折最高風險 10% 警示規則，並保留 pooled OOF AP。
- 用病人為抽樣單位、折內分層的配對 bootstrap 報 AP 差和 95% CI；另報每折差值與外部特徵覆蓋率。以同架構、同特徵集合只差一組外部先驗的消融，判斷 HCUPnet／CMS 是否真的比 MEPS 增益大。
- 解釋 HCUPnet／CMS 特徵的 SHAP 排名與方向分布，但是否改善仍由測試折 AP 差判斷。CMS 的風險標準化醫院率、HCUPnet 的全國粗率和 UCI 的個體風險不應直接當同一尺度的機率平均；讓模型以特徵方式使用，或在訓練折內校準。
- 先檢查 HCUPnet 是否有完整交叉表、足夠疾病覆蓋與分母，再投入五折訓練。若交叉表不可取得，退回「疾病群組率」實驗並如實命名。若要真正用外部逐筆再住院標籤訓練，可考慮 HCUP NRD 原始資料；HCUP FAQ 說 NRD 需透過 Central Distributor 購買。[HCUP FAQ](https://hcup-us.ahrq.gov/tech_assist/faq.jsp)。

## 目前結論

HCUPnet 比 CMS 公開品質表更符合「不限定糖尿病、按疾病群組與年齡借用真實 30 天再住院訊號」的想法；CMS 可以提供較窄疾病範圍的輔助訊號。**目前沒有 HCUPnet／CMS 的五折結果，不能判定是否優於 MEPS、NHANES 或無外部資料模型。**
