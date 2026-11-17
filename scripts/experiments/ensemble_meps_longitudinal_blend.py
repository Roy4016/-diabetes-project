#!/usr/bin/env python3
"""Average OOF probabilities from pre-specified longitudinal blend seeds."""

import argparse
import sys

from _paths import PROJECT_ROOT, result_dir

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.longitudinal_ensemble import run_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=[
        result_dir("meps_longitudinal_blend_results"),
        result_dir("meps_longitudinal_blend_seed2024"),
        result_dir("meps_longitudinal_blend_seed3407"),
    ])
    parser.add_argument("--outdir", default=result_dir("meps_longitudinal_blend_ensemble_results"))
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    summary, pooled, bootstrap, external_signal = run_ensemble(
        args.inputs, args.outdir, args.seed, args.bootstrap,
    )
    print("\nFold summary\n", summary.round(6).to_string(index=False))
    print("\nPooled metrics\n", pooled.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))
    print("\nMatched external-signal bootstrap\n", external_signal.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
