#!/usr/bin/env python3
"""Download real NHANES diabetic respondents with past-year hospitalization answers."""

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from _paths import PROJECT_ROOT


CYCLES = {"2015-2016": "I", "2017-2018": "J"}


def read_xport(path):
    with path.open("rb") as stream:
        header = stream.read(80)
    if not header.startswith(b"HEADER RECORD"):
        raise ValueError(f"{path} is not a SAS XPORT file (received {header[:40]!r})")
    return pd.read_sas(path, format="xport")


def fetch_cycle(raw_dir, cycle, suffix):
    start = cycle[:4]
    tables = {}
    for name in ("DEMO", "DIQ", "HUQ"):
        filename = f"{name}_{suffix}.xpt"
        path = raw_dir / filename
        if not path.exists():
            url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{filename}"
            print(f"[{cycle}] downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "diabetes-project/1.0"})
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                    final_url = response.geturl()
                    content_type = response.headers.get("Content-Type", "unknown")
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                try:
                    table = read_xport(temporary)
                except ValueError as exc:
                    raise ValueError(
                        f"NHANES returned an invalid file from {final_url} "
                        f"(Content-Type: {content_type}). Check the CDC download link."
                    ) from exc
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            table = read_xport(path)
        tables[name] = table
    merged = tables["DEMO"].merge(tables["DIQ"], on="SEQN", validate="one_to_one")
    merged = merged.merge(tables["HUQ"], on="SEQN", validate="one_to_one")
    required = {"DIQ010", "HUQ071", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTINT2YR"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"NHANES {cycle} missing {sorted(missing)}")
    selected = merged.loc[merged.DIQ010.eq(1) & merged.HUQ071.isin([1, 2])].copy()
    selected["cycle"] = cycle
    selected["year"] = int(start)
    selected["hospitalized_past_year"] = selected.HUQ071.eq(1).astype(int)
    selected["age"] = pd.to_numeric(selected.RIDAGEYR, errors="coerce")
    selected["male"] = selected.RIAGENDR.map({1: 1., 2: 0.})
    selected["race_common"] = selected.RIDRETH3.map({
        1: "Hispanic", 2: "Hispanic", 3: "Caucasian",
        4: "AfricanAmerican", 6: "Asian", 7: "Other",
    })
    selected["person_weight"] = pd.to_numeric(selected.WTINT2YR, errors="coerce")
    return selected[["cycle", "year", "SEQN", "age", "male", "race_common",
                     "hospitalized_past_year", "person_weight"]].dropna(
        subset=["age", "male", "race_common", "person_weight"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/nhanes/2015_2018")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/nhanes/diabetes_hospitalization_2015_2018.csv")
    parser.add_argument("--cycles", nargs="+", choices=CYCLES, default=list(CYCLES))
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat([fetch_cycle(args.raw_dir, cycle, CYCLES[cycle]) for cycle in args.cycles], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"Saved {len(data):,} real NHANES diabetic respondents; "
          f"past-year hospitalization rate={data.hospitalized_past_year.mean():.4f} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
