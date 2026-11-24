# Specialized experiments

## HCUPnet／CMS 外部再住院資料

CMS 公開的醫院級與全國級再住院表可用 `python scripts/experiments/download_cms_readmissions.py` 下載至 `data/raw/cms/`。執行 `python scripts/experiments/prepare_cms_readmission_prior.py` 可產生四個可對應 UCI ICD-9 CCS 主要診斷的 CMS 群組比率及覆蓋率稽核。HCUPnet 疾病 × 年齡統計需在網站匯出 Excel 後存放於 `data/raw/hcupnet/`。資料粒度、來源與五折接法見 [可行性文件](../../docs/hcupnet_cms_external_feasibility.md)。

## 既有實驗

These scripts contain the existing CDC, MEPS, imbalance, and CCS experiments.
Their historical outputs are under `../../results`. Run them from the project root
and pass explicit paths from `data/raw` and `data/processed`; older default paths
were written for the previous flat directory layout.

Current five-fold TabPFN 3.5 and corrected inner-CV commands, metrics, split
audit, and continuation instructions are in `../../docs/pending_experiments.md`.

# Current five-fold LR, RF, and XGBoost recheck

The historical LR, RF, standard XGBoost, class-weighted XGBoost, and four
SMOTENC XGBoost runs used a single holdout. Recheck all ten variants on the
exact 99,343-encounter, patient-disjoint five-fold split used by the recent
CCS/MEPS experiments:

```bash
cd /media/synology/G/314554044/diabetes_project
CUDA_VISIBLE_DEVICES=5 python scripts/experiments/run_classic_5fold.py \
  --reference-oof results/meps_ccs_group_hgb_seed777/oof_all.csv \
  --outdir results/classic_5fold_recheck_seed777 \
  --max-folds 5 --device cuda --seed 777
```

The script checks encounter IDs, labels, patient IDs, and folds against the
reference file. It selects tree count and F1 threshold on a patient-disjoint
inner validation set. SMOTENC is applied only to training rows, and the final
model is refit on the full outer training fold. LR and RF use CPU; XGBoost uses
the selected GPU. Outputs are `fold_metrics.csv`,
`summary.csv`, `oof_long.csv`, `selections.csv`, and `run.json`. `PR_AUC_AP`
is scikit-learn average precision. The `val_f1_optimal` rows contain all six
requested metrics; `default_0.5` rows are also included.
