#!/usr/bin/env python3
"""Nested blend of Clinical XGBoost and longitudinal MEPS risk XGBoost."""

import argparse
import sys

from _paths import PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.longitudinal_blend import LongitudinalBlendConfig, run_longitudinal_blend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", default=str(PROJECT_ROOT / "data/processed/uci_5fold"))
    parser.add_argument("--uci", default=UCI_RAW)
    parser.add_argument("--meps", default=str(PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv"))
    parser.add_argument("--outdir", default=result_dir("meps_longitudinal_blend_results"))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--uci-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--meps-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-meps", type=float, default=.2)
    parser.add_argument("--lambda-values", nargs="+", type=float)
    parser.add_argument("--expense-weight", type=float, default=.2)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-early-stopping", type=int, default=25)
    parser.add_argument("--max-external-weight", type=float, default=.6)
    parser.add_argument("--blend-steps", type=int, default=13)
    parser.add_argument("--min-blend-gain", type=float, default=0.)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = LongitudinalBlendConfig(
        folds_dir=args.folds_dir, uci_csv=args.uci, meps_csv=args.meps, output_dir=args.outdir,
        max_folds=args.max_folds, seed=args.seed, svd_components=args.svd_components,
        embedding_dim=args.embedding_dim, pretrain_epochs=args.pretrain_epochs,
        uci_epochs=args.uci_epochs, patience=args.patience, batch_size=args.batch_size,
        meps_batch_size=args.meps_batch_size, learning_rate=args.learning_rate,
        lambda_meps=args.lambda_meps, lambda_values=tuple(args.lambda_values or ()),
        expense_weight=args.expense_weight,
        xgb_estimators=args.xgb_estimators, xgb_early_stopping=args.xgb_early_stopping,
        max_external_weight=args.max_external_weight, blend_steps=args.blend_steps,
        min_blend_gain=args.min_blend_gain,
        bootstrap=args.bootstrap, device=args.device,
    )
    summary, bootstrap, external_signal = run_longitudinal_blend(config)
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))
    print("\nMatched external-signal bootstrap\n", external_signal.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
