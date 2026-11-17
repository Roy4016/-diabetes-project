#!/usr/bin/env python3
"""Append aligned longitudinal MEPS representations to Clinical XGBoost."""

import argparse
import sys

from _paths import PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.longitudinal_xgb import LongitudinalXGBConfig, VARIANTS, run_longitudinal_xgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", default=str(PROJECT_ROOT / "data/processed/uci_5fold"))
    parser.add_argument("--uci", default=UCI_RAW)
    parser.add_argument("--meps", default=str(PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv"))
    parser.add_argument("--encoder-dir", default=result_dir("meps_longitudinal_aligned_results"))
    parser.add_argument("--outdir", default=result_dir("meps_longitudinal_xgb_results"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--meps-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expense-weight", type=float, default=.2)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-early-stopping", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = LongitudinalXGBConfig(
        folds_dir=args.folds_dir, uci_csv=args.uci, meps_csv=args.meps,
        encoder_dir=args.encoder_dir, output_dir=args.outdir, variants=tuple(args.variants),
        max_folds=args.max_folds, seed=args.seed, embedding_dim=args.embedding_dim,
        pretrain_epochs=args.pretrain_epochs, patience=args.patience,
        meps_batch_size=args.meps_batch_size, learning_rate=args.learning_rate,
        expense_weight=args.expense_weight, xgb_estimators=args.xgb_estimators,
        xgb_early_stopping=args.xgb_early_stopping, bootstrap=args.bootstrap, device=args.device,
    )
    summary, bootstrap, external_signal = run_longitudinal_xgb(config)
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))
    print("\nMatched external-signal bootstrap\n", external_signal.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
