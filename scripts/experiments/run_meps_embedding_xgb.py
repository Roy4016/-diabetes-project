#!/usr/bin/env python3
"""Train MEPS-supervised embeddings and append them to Clinical XGBoost."""

import argparse
import sys
from pathlib import Path

from _paths import PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.external_embedding import EmbeddingExperimentConfig, VARIANTS, run_embedding_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", default=str(PROJECT_ROOT / "data/processed/uci_5fold"))
    parser.add_argument("--uci", default=UCI_RAW)
    parser.add_argument("--meps", default=str(PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv"))
    parser.add_argument("--outdir", default=result_dir("meps_embedding_xgb_results"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--multitask-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--meps-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-meps", type=float, default=0.2)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-early-stopping", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = EmbeddingExperimentConfig(
        folds_dir=args.folds_dir, uci_csv=args.uci, meps_csv=args.meps,
        output_dir=args.outdir, variants=tuple(args.variants), max_folds=args.max_folds,
        seed=args.seed, embedding_dim=args.embedding_dim, pretrain_epochs=args.pretrain_epochs,
        multitask_epochs=args.multitask_epochs, patience=args.patience,
        batch_size=args.batch_size, meps_batch_size=args.meps_batch_size,
        learning_rate=args.learning_rate, lambda_meps=args.lambda_meps,
        xgb_estimators=args.xgb_estimators, xgb_early_stopping=args.xgb_early_stopping,
        bootstrap=args.bootstrap, device=args.device,
    )
    summary, bootstrap = run_embedding_experiment(config)
    print("\nSummary\n", summary.round(6).to_string(index=False))
    if not bootstrap.empty:
        print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
