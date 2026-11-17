#!/usr/bin/env python3
"""Create the processed 5-fold patient-disjoint dataset used by experiments."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.paths import DEFAULT_FOLDS, UCI_RAW
from diabetes_project.preprocessing import preprocess_uci_5fold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=UCI_RAW, help="Raw UCI diabetic_data.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--no-clinical-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = preprocess_uci_5fold(
        input_path=args.input,
        output_dir=args.output,
        n_splits=args.folds,
        seed=args.seed,
        validation_size=args.validation_size,
        clinical_features=not args.no_clinical_features,
        overwrite=args.overwrite,
    )
    print(
        f"Created {summary['folds']} patient-disjoint folds at {args.output.resolve()}\n"
        f"Rows={summary['rows']:,}; patients={summary['patients']:,}; "
        f"positive rate={summary['positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()

