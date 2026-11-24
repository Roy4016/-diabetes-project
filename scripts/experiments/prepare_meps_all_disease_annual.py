#!/usr/bin/env python3
"""Build all-disease MEPS person-year bridge using the diabetic bridge rules."""

import argparse
from pathlib import Path

import pandas as pd

from _paths import PROJECT_ROOT
from prepare_meps import FILES, process_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data/raw/meps")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/meps/meps_all_disease_bridge8_2008_2012.csv")
    args = parser.parse_args()
    frames = [process_year(args.input_root, year, FILES[year], diabetes_only=False)
              for year in range(2008, 2013)]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["year", "person_key"]).any():
        raise ValueError("Duplicate person-year identifiers in full MEPS bridge")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Saved {len(combined):,} person-years, {combined.person_key.nunique():,} people to {args.output}", flush=True)


if __name__ == "__main__":
    main()
