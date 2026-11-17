# Code audit and consolidation

The review found repeated implementations in three main areas:

1. `StratifiedGroupKFold` plus an inner `GroupShuffleSplit`, followed by manual
   patient-overlap assertions.
2. Training-only categorical encoding and numeric missing-value handling.
3. F1 threshold selection, binary metrics, calibration/ROC/PR figures, prediction
   exports, and nearly identical LR/RF/XGBoost scripts.

These are now centralized as follows:

| Concern | Shared module |
|---|---|
| Raw UCI cohort and feature exclusions | `src/diabetes_project/data.py` |
| Clinical diagnosis/utilization features | `src/diabetes_project/features.py` |
| Patient-disjoint outer and inner folds | `src/diabetes_project/splitting.py` |
| Fold materialization and fitted preprocessors | `src/diabetes_project/preprocessing.py` |
| Thresholds and metrics | `src/diabetes_project/metrics.py` |
| ROC, PR, and calibration plots | `src/diabetes_project/plots.py` |
| LR, RF, and XGBoost orchestration | `src/diabetes_project/baselines.py` |
| MEPS multi-task embeddings and XGBoost controls | `src/diabetes_project/external_embedding.py` |
| Inner-fold MEPS embedding hyperparameter selection | `src/diabetes_project/external_tuning.py` |

`preprocess.py` and `train_baselines.py` are thin command-line entry points over
the package. The previous per-method scripts remain in `archive/legacy_scripts`
only to preserve provenance for prior results.

Specialized CDC, MEPS, CCS, and imbalance experiments remain separate because
their scientific procedures differ. Their common dataset/result defaults now
come from `scripts/experiments/_paths.py`, so they follow the reorganized layout.
