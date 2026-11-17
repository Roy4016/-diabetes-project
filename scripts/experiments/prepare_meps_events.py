#!/usr/bin/env python3
"""Create exact 30-day MEPS event outcomes for diabetic inpatient stays."""

import argparse
import json
import sys

from _paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.meps_events import build_event_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", default=str(PROJECT_ROOT / "data/raw/meps/events_2008_2024"))
    parser.add_argument("--annual", default=str(PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "data/processed/meps_events_2008_2012"))
    args = parser.parse_args()
    _, _, summary = build_event_dataset(args.event_root, args.annual, args.outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
