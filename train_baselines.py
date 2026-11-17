#!/usr/bin/env python3
"""Train one or more baseline model families from a processed fold."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.baselines import run_baselines
from diabetes_project.paths import DEFAULT_FOLDS, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=["lr", "rf", "xgb"], default=["lr", "rf", "xgb"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    table = run_baselines(
        args.data_dir / f"fold_{args.fold}", args.output / f"fold_{args.fold}",
        args.models, seed=args.seed, n_estimators=args.n_estimators,
        max_depth=args.max_depth, max_iter=args.max_iter, n_jobs=args.n_jobs,
    )
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

