#!/usr/bin/env python3
"""Transfer exact MEPS 30-day event risk into Clinical XGBoost."""

import argparse
import sys
from _paths import PROJECT_ROOT, UCI_RAW, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from diabetes_project.event_transfer import EventTransferConfig, VARIANTS, run_event_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", default=str(PROJECT_ROOT / "data/processed/uci_5fold"))
    parser.add_argument("--uci", default=UCI_RAW)
    parser.add_argument("--events", default=str(PROJECT_ROOT / "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv"))
    parser.add_argument("--outdir", default=result_dir("meps_event_transfer_results"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--er-weight", type=float, default=.2)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-early-stopping", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = EventTransferConfig(
        folds_dir=args.folds_dir, uci_csv=args.uci, event_csv=args.events,
        output_dir=args.outdir, variants=tuple(args.variants), max_folds=args.max_folds,
        seed=args.seed, embedding_dim=args.embedding_dim, pretrain_epochs=args.pretrain_epochs,
        patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
        er_weight=args.er_weight, xgb_estimators=args.xgb_estimators,
        xgb_early_stopping=args.xgb_early_stopping, bootstrap=args.bootstrap, device=args.device,
    )
    summary, bootstrap, external_signal = run_event_transfer(config)
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))
    print("\nMatched external-signal bootstrap\n", external_signal.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
