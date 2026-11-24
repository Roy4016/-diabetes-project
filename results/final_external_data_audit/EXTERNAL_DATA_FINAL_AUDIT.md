# 最終簡報：External Data 實作稽核

範圍限於 `run_cms_meps_cdc_catboost.py`、`run_meps_all_disease_30d_transfer.py` 及其必要函式。以下說明程式實際產生的特徵與資料流；**未重新訓練或核算分數**。文中路徑均相對於專案根目錄。

## 1. 接法總覽

| 接法 | 外部資料與標籤 | 加入 UCI 的欄位 | 接收模型／融合 |
|---|---|---|---|
| CDC | BRFSS 2015 糖尿病受訪者；自評健康「尚可／差」 | `cdc_poor_health` | 原始 46 欄 CatBoost；單獨或與其他來源併用 |
| MEPS Statistical | 2008–2012 糖尿病住院事件；30 天再住院、急診再訪 | `meps_ccs_readmit_mean`, `meps_ccs_revisit_mean`, `meps_ccs_readmit_max`, `meps_ccs_revisit_max` | 三來源腳本加到原始 46 欄 CatBoost；舊 `ccs_meps_catboost_raw_50` 加到 HGB |
| CMS | 四項 CMS 全國風險標準化 30 天再住院率 | `cms_readmission_rate`, `cms_mapped` | 原始 46 欄 CatBoost；單獨或與其他來源併用 |
| MEPS Teacher | 2008–2012 全疾病住院事件；由事件日期推得的 30 天再住院 | `meps_30d_teacher_risk`, `meps_ccs_log_support` | 原始 ICD-9＋CCS 的 CatBoost，與既存 CCS HGB 以 50/50 融合 |
| MEPS Joint Training | 同一全疾病事件及 UCI 住院紀錄，兩個任務各有標籤 | **無固定外部欄位**；以共享網路參數轉移 | UCI 神經網路；另可與既存 CatBoost 以 50/50 融合 |

三來源腳本的實驗欄位清單見 `scripts/experiments/run_cms_meps_cdc_catboost.py:39-48`；模型輸入與融合見同檔 `:149-207`。Teacher 與 joint 的接法見 `scripts/experiments/run_meps_all_disease_30d_transfer.py:75-163`。

## 2. CDC BRFSS：健康狀態代理分數

**來源及輸入。** 預設檔為 `data/raw/cdc/diabetes_012_health_indicators_BRFSS2015.csv`。讀入 `Age`, `Sex`, `HighBP`, `HighChol`, `GenHlth`, `Diabetes_012`；保留 `Diabetes_012 == 2` 的受訪者。`GenHlth` 的 4、5 定義 `poor_health=1`，1–3 定義 0。CDC 年齡類別換成年齡近似中點，13 類（80 歲以上）設為 85；來源缺少四個 bridge 欄位或 `GenHlth` 的列被排除，性別／血壓／膽固醇指標限 0/1。來源：`scripts/experiments/run_cdc_xgb.py:28-88`；糖尿病篩選：`scripts/experiments/run_cms_meps_cdc_catboost.py:63-71,247`。

| CDC bridge 欄位 | UCI 對應及轉換 | 限制 |
|---|---|---|
| `bridge_age` | `age` 十歲區間取中點；CDC `Age` 類別亦取近似中點 | 年齡不是精確值 |
| `bridge_sex` | `gender`: female→0、male→1；未知為缺值 | 二元近似 |
| `bridge_high_bp` | `diag_1/2/3` 任一 ICD-9 三位前綴 401–405→1，否則 0 | 已記錄診斷的代理值，不是病人完整高血壓史 |
| `bridge_high_chol` | `diag_1/2/3` 任一 ICD-9 三位前綴 272→1，否則 0 | 同上，與 CDC 問卷測量方式不同 |

UCI 映射程式：`scripts/experiments/run_cdc_xgb.py:90-181`。CDC 模型為 CPU XGBoost 二元分類器，300 棵樹、深度 3、學習率 0.05、`min_child_weight=5`、`reg_lambda=2`；先以 seed 777 做 80/20 分層驗證，最後以全部合格 CDC 受訪者重訓一次，再對 UCI 預測 `P(GenHlth≥4)`。產出的唯一 CDC 欄位是 `cdc_poor_health`；沒有在此主實驗加入 DiffWalk、embedding 或其他健康分數。模型**不是**預測 30 天再住院，也不是每個 UCI 外層折各訓一次。UCI bridge 若四欄全缺會報錯；個別缺值未在這段程式另行填補，由 XGBoost 處理。來源：`scripts/experiments/run_cms_meps_cdc_catboost.py:63-101,122-133`。

## 3. MEPS Statistical：事件 CCS 群組統計

預設輸入為 `data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv` 與 `meps_inpatient_event_ccs.csv`；它們由 MEPS 2008–2012 住院／急診事件及條件－事件連結檔建立。標籤依同一人下一次住院／急診日期與本次出院日期計算，日期或追蹤期不確定的情況被排除；條件經 CCS 編碼後以 `year, person_key, event_id` 接到**同一事件**。來源：`scripts/experiments/run_cms_meps_cdc_catboost.py:113-120,245-246`、`src/diabetes_project/meps_events.py:94-171`、`scripts/experiments/prepare_meps_event_ccs.py:44-102`。

只保留再住院與急診標籤都為 0/1 的事件；同一事件同一 CCS 僅計一次。對每個至少 30 筆的 CCS 群組，使用 `α=50` 平滑：

`group_readmit = (群組 30 天再住院正例數 + 50 × 全事件再住院率) / (群組事件數 + 50)`；急診再訪率同式。此處**沒有**用 MEPS 問卷權重。程式內部 `next_er` 變數實際存的是再住院率，`next_ip` 實際存的是急診再訪率；輸出欄名 `readmit`／`revisit` 才與真實標籤一致。來源：`scripts/experiments/run_meps_ccs_group_hgb.py:168-199`、`scripts/experiments/run_cms_meps_cdc_catboost.py:255-256`。

UCI 的 `diag_1/2/3` 先透過 AHRQ 2015 單層 CCS 表轉為 `ccs_diag_1/2/3`；匹配採精確 ICD-9 碼或所有後代一致的唯一前綴，缺碼／不明確前綴另標記。每個診斷位置查群組再住院與急診率，找不到或群組未達 30 筆時用相應**全事件平均率**；三個位置分別取平均與最大，形成四個 `EVENT_STATS` 欄位。來源：`scripts/experiments/run_meps_ccs_group_hgb.py:23-24,114-121,202-215`、`scripts/experiments/train_xgboost_ahrq_ccs_prefix.py:34-51,90-118`。

**重要差異：** 三來源腳本中的 `meps` 是「原始 46 欄 CatBoost＋上述四欄」，沒有 HGB 融合。舊 `ccs_meps_catboost_raw_50` 則是「原始 46 欄＋三個 CCS＋四個 MEPS 統計欄位」訓練 HGB，再與**沒有 MEPS 欄位**的 `catboost_raw` 平均：`0.5 × ccs_meps_group_hgb + 0.5 × catboost_raw`。兩者不能視為同一種模型。來源：`scripts/experiments/run_cms_meps_cdc_catboost.py:40-46,181-207`、`scripts/experiments/run_paper_fair_comparison.py:138-179,376-379`。

## 4. CMS：全國疾病別彙總率

預設已處理表 `data/processed/cms/cms_readmission_ccs_prior.csv` 由 `unplanned_hospital_visits_national.csv` 及 `unplanned_hospital_visits_hospital.csv` 準備，選取 `READM_30_AMI`, `READM_30_HF`, `READM_30_PN`, `READM_30_COPD`，分別近似映射 `CCS_100`, `CCS_108`, `CCS_122`, `CCS_127`。模型欄位使用的是各措施唯一的 **`national_rate`**，有效範圍 0–100，除以 100 後成比例；醫院層級 `score` 僅在先驗表中用來計數可得值與計算中位數，**沒有**拿來加權、平均或當模型特徵。來源：`scripts/experiments/prepare_cms_readmission_prior.py:20-105`、`scripts/experiments/run_cms_group_prior_5fold.py:49-60`。

UCI 僅以主診斷 `ccs_diag_1` 對四類查表：匹配則 `cms_readmission_rate=國家率/100`、`cms_mapped=1`；否則分別為 `-1`、`0`。`diag_2/3` 不參與 CMS 匹配。這是**疾病層級的背景率**，不是該 UCI 病人的再住院機率，也不代表符合 CMS 措施的全部資格。來源：`scripts/experiments/run_cms_meps_cdc_catboost.py:103-110`、`scripts/experiments/prepare_cms_readmission_prior.py:20-27,62-94`。

## 5. 簡報模型矩陣與融合

下表中的「原始 CatBoost」為 46 個出院時特徵，保留 `diag_1/2/3` 原始 ICD-9；類別缺值轉 `__MISSING__`。三來源腳本的 CatBoost 每折僅用外層訓練資料訓練，沿用既有、在訓練資料內選出的 CatBoost 參數。`ccs_only_hgb` 使用 46 個原始欄位加三個 CCS 類別。來源：`src/diabetes_project/paper_comparison.py:33-84`、`scripts/experiments/run_cms_meps_cdc_catboost.py:149-207`、`scripts/experiments/run_dual_external_meps_nhanes.py:53-64`。

| 輸出名稱 | HGB 輸入 | CatBoost 輸入 | 融合 |
|---|---|---|---|
| `catboost_raw` | 無 | 原始 46 欄 | 無 |
| `cms_catboost_raw` | 無 | 原始 46 欄＋CMS 2 欄 | 無；由先前 CMS 五折腳本載入 |
| `meps` | 無 | 原始 46 欄＋MEPS Statistical 4 欄 | 無 |
| `cdc` | 無 | 原始 46 欄＋CDC 1 欄 | 無 |
| `cms_meps` | 無 | 原始 46 欄＋CMS 2＋MEPS 4 | 無 |
| `cms_cdc` | 無 | 原始 46 欄＋CMS 2＋CDC 1 | 無 |
| `meps_cdc` | 無 | 原始 46 欄＋MEPS 4＋CDC 1 | 無 |
| `cms_meps_cdc` | 無 | 原始 46 欄＋CMS 2＋MEPS 4＋CDC 1 | 無 |
| `cms_meps_cdc_hgb50` | **既存** `ccs_only_hgb`（46 原始＋CCS 3） | 同上 | `0.5 × ccs_only_hgb + 0.5 × cms_meps_cdc` |
| `ccs_only_hgb_catboost_raw_50` | **既存** `ccs_only_hgb` | **既存** `catboost_raw` | `0.5 × ccs_only_hgb + 0.5 × catboost_raw` |
| `ccs_meps_catboost_raw_50` | 46 原始＋CCS 3＋MEPS Statistical 4 | 既存原始 46 欄 | `0.5 × ccs_meps_group_hgb + 0.5 × catboost_raw` |

融合直接平均 `predict_proba`／既存 OOF 機率，這些腳本沒有在融合前作 Platt、isotonic 等校準。`cms_catboost_raw` 的實際新增欄位見 `scripts/experiments/run_cms_group_prior_5fold.py:76-97,123-147`；其餘行的特徵選取及公式見 `scripts/experiments/run_cms_meps_cdc_catboost.py:40-48,169-207`。

## 6. 新 MEPS 全疾病 Teacher 與 Joint Training

### 6.1 來源、標籤與映射

全疾病來源是 2008–2012 MEPS 住院與急診事件檔，以及**不按糖尿病篩選**的年度個人資料。`readmission_30d=1` 表示同一人出院後第 **1–30 天**有下一次住院；不確定日期及 2012 年底前無法完成 30 天追蹤者不當成確定陰性。此標籤由事件日期推得，不是 MEPS 原始標籤；其邊界與 UCI `<30` 類別未必完全一致。來源：`scripts/experiments/prepare_meps_all_disease_annual.py:10-28`、`scripts/experiments/prepare_meps_all_disease_30d.py:27-102`。

MEPS 與 UCI 的共同欄位為：`age`（0–100 歲除以 100）、`male`、`race_Caucasian`, `race_AfricanAmerican`, `race_Hispanic`, `race_Asian`, `race_Other`、`length_of_stay`（截到 0–30 天再除以 30）、以及最常見至多 64 個 CCS 的 `group_CCS_*` 二元欄位。UCI 的 group 為三個出院診斷任一匹配；MEPS 的 group 來自與該住院事件連結的 CCS。年齡缺值填 65、性別缺值填 0.5、住院日數缺值填 0；未知 race 對應欄位皆為 0。MEPS 額外私有欄位為 `emergency_admission`, `any_operation`, `discharge_medicine`，缺值填 0.5。來源：`src/diabetes_project/meps_30d_transfer.py:18-69`、`src/diabetes_project/harmonized_meps.py:15,39-55`。

### 6.2 Teacher-Based Feature Transfer

Teacher 是 CatBoost（二元 Logloss、深度 4、預設 400 棵、學習率 0.035、L2=10）。先以 **MEPS 人員分組 80/20** 檢查，再使用全部 MEPS 訓練一次；**不在每個 UCI 折重新訓練 Teacher**。它只吃上述可對齊的共同欄位，對每筆 UCI 產生 `meps_30d_teacher_risk`。第二欄 `meps_ccs_log_support = log(1 + Σ UCI 所匹配各 CCS 群組在 MEPS 的事件數)`，是**樣本支持度而非再住院率**。因此這條路徑**沒有**加入前述四個 `meps_ccs_*_mean/max` 統計欄位。來源：`scripts/experiments/run_meps_30d_joint_transfer.py:127-154`、`src/diabetes_project/meps_30d_transfer.py:64-69`、`scripts/experiments/run_meps_all_disease_30d_transfer.py:83-110`。

`all_disease_meps_teacher_catboost` 使用原始 ICD-9 46 欄＋`ccs_diag_1/2/3`＋上述兩個 Teacher 欄位，在每個 UCI 外層訓練折重訓 CatBoost，採既存基線每折選好的深度與樹數。`all_disease_meps_teacher_hgb50 = 0.5 × 既存 ccs_only_hgb + 0.5 × all_disease_meps_teacher_catboost`；HGB **沒有**接收 Teacher 欄位。`diabetes_meps_teacher_hgb50` 以**已保存的糖尿病限定 Teacher 分數及 support**，搭配同一 CatBoost 基礎欄位、折與參數重訓，形成配對對照；程式先驗證 encounter／patient／label／fold 完全相同。來源：`scripts/experiments/run_meps_all_disease_30d_transfer.py:38-72,75-137`、`scripts/experiments/run_ccs_catboost_5fold.py:37-57`。

### 6.3 Joint Training

UCI 分支輸入共同欄位，以及原始 ICD-9 的 46 欄經**該外層訓練折**類別填值＋one-hot、數值中位數填值＋標準化後的私有表示；MEPS 分支輸入共同欄位及三個事件私有欄位。共享編碼器為 `common_dim→48→24`；UCI 私有層 `uci_dim→64`（dropout 0.10），MEPS 私有層 `3→12`；UCI head `88→32→1`、MEPS head `36→16→1`。來源：`scripts/experiments/run_meps_30d_joint_transfer.py:34-81`。

每個 UCI 外層折從頭訓練：每步以該折 UCI 訓練資料算 BCE；`all_disease_joint_meps_30d` 每四步再抽一批 MEPS 事件，加上 `0.3 × BCE_MEPS`（皆為預設參數，12 epochs）。`uci_only_nn` 使用同一架構與 UCI 訓練程序，但不加 MEPS loss。推論只走 UCI head 並取 sigmoid；可另外與**既存 CatBoost**機率 50/50 平均。Joint 模型**沒有**產生固定的 `meps_30d_teacher_risk` 欄位。Joint 訓練須在實驗指令加 `--include-joint`；否則只跑 Teacher 路徑。來源：`scripts/experiments/run_meps_30d_joint_transfer.py:83-124`、`scripts/experiments/run_meps_all_disease_30d_transfer.py:137-164,205-211`。

**三種 MEPS 接法不能互稱：** Statistical 是 CCS 群組**平滑觀察率**；Teacher 是 MEPS 監督模型在 UCI 上產生的**風險分數＋群組支持度**；Joint 是以 MEPS loss 更新**共享神經網路參數**，沒有靜態外部欄位。全疾病 Teacher/Joint 亦不等於三來源腳本中的糖尿病限定 MEPS Statistical。來源如上。

## 7. 五折、洩漏與適用限制

| 檢查 | 結論與程式依據 |
|---|---|
| UCI 分折與基線 | `load_reference` 要求 99,343 筆、五折、同一患者只在一折；三來源模型逐 encounter 核對 patient、label、fold；全疾病腳本另核對最佳 CCS HGB＋CCS CatBoost OOF。`scripts/experiments/run_dual_external_meps_nhanes.py:40-64`；`scripts/experiments/run_cms_meps_cdc_catboost.py:51-60,149-175`；`scripts/experiments/run_ccs_ensemble_imbalance_5fold.py:38-52`。 |
| Seed | 三來源腳本預設 777，但允許其他 `--seed` 搭配**既存**五折 OOF，故呈報時應檢查實際 `audit.json`；全疾病腳本明確限定 `seed==777`。`scripts/experiments/run_cms_meps_cdc_catboost.py:240-261`；`scripts/experiments/run_meps_all_disease_30d_transfer.py:75-80`。 |
| UCI 測試標籤 | CDC 模型只用 CDC `poor_health`；MEPS 群組統計／Teacher 只用 MEPS 標籤；CMS 是固定外部率。CatBoost/Joint 以 `Fold != test_fold` 訓練。靜態程式**未見** UCI 外層測試標籤進入外部特徵計算。`scripts/experiments/run_cms_meps_cdc_catboost.py:63-123,177-191`；`scripts/experiments/run_meps_all_disease_30d_transfer.py:83-145`。 |
| 前處理 | Joint 的 UCI one-hot、填值、標準化只對外層訓練折 `.fit()`；三來源 CatBoost 的固定外部彙總不依賴 UCI 標籤。`scripts/experiments/run_meps_30d_joint_transfer.py:62-81`。 |
| 預測時點 | UCI 診斷、出院處置與住院日數是出院時資訊；所述方法應定位為**出院時**預測，不能聲稱入院時即可使用。 |

**須在簡報保留的限制：** CDC 自評健康與住院病歷的測量方式不同；MEPS 來源與 UCI 的族群和標籤邊界不同；CMS 僅四個近似 CCS 概念，未匹配時用 `-1/0`，且全國率不是個人風險；三來源腳本的 CatBoost 原始 46 欄基線與全疾病 Teacher 腳本的「原始 46 欄＋CCS 3 欄」基線不同，跨腳本不可把 AP 差值直接當成單一外部來源效益。這些是方法上的限制或可能疑慮，**不是已證實的性能因果原因**。

## 8. Four concise English slide drafts

### Slide 1 — CDC Integration

- BRFSS 2015 diabetic respondents train a poor-health XGBoost proxy (`GenHlth` fair/poor).
- Align UCI age and sex with ICD-9 proxies for hypertension and hypercholesterolemia.
- Append one feature, `cdc_poor_health`, to the discharge CatBoost model.
- This score estimates self-rated health, not 30-day readmission.

### Slide 2 — MEPS Statistical Integration

- Link 2008–2012 diabetic inpatient events to their event-specific CCS diagnoses.
- Estimate smoothed CCS-group 30-day readmission and ER-revisit rates.
- Map UCI `diag_1/2/3` to CCS; append mean and maximum rates across the three diagnoses.
- Unmatched groups receive the corresponding MEPS global event rate.

### Slide 3 — CMS Integration

- Use four CMS national risk-standardized 30-day readmission measures: AMI, HF, pneumonia and COPD.
- Approximate each measure with one AHRQ CCS group; match UCI principal diagnosis only.
- Append `cms_readmission_rate` and `cms_mapped` to CatBoost.
- National disease-level rates provide context, not individualized probabilities.

### Slide 4 — MEPS All-Disease Teacher Transfer

- Derive exact-date 30-day readmission labels from all-disease MEPS inpatient events.
- Train one MEPS CatBoost teacher on aligned age, sex, race, length of stay and CCS-group indicators.
- Add `meps_30d_teacher_risk` and `meps_ccs_log_support` to raw ICD-9 plus CCS CatBoost.
- Average the augmented CatBoost probability with saved CCS HGB at 50/50; compare with the diabetic-only teacher on identical UCI folds.
