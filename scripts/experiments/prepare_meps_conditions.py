#!/usr/bin/env python3
"""Download 2008-2012 MEPS Medical Conditions PUFs and extract person-year CCS."""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from _paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts/experiments"))
from prepare_meps import make_person_key


PUFS = {2008: 120, 2009: 128, 2010: 137, 2011: 146, 2012: 154}


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".ssp", ".xpt", ".xport"))]
        if len(names) != 1:
            raise ValueError(f"Expected one SAS transport file in {path}: {archive.namelist()}")
        frame = pd.read_sas(io.BytesIO(archive.read(names[0])), format="xport", encoding="latin-1")
    frame.columns = frame.columns.str.upper()
    missing = {"PANEL", "DUPERSID", "CCCODEX"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/meps/conditions_2008_2012")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/meps/meps_conditions_ccs_2008_2012.csv")
    parser.add_argument("--annual", type=Path, default=PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv")
    parser.add_argument("--years", nargs="+", type=int, default=list(PUFS), choices=PUFS)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(args.annual, usecols=["year", "person_key"], dtype={"person_key": str})
    allowed = set(zip(annual.year.astype(int), annual.person_key))
    parts = []
    for year in args.years:
        puf = PUFS[year]
        path = args.raw_dir / f"h{puf}ssp.zip"
        if not path.exists():
            url = f"https://meps.ahrq.gov/mepsweb/data_files/pufs/h{puf}ssp.zip"
            print(f"[{year}] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path)
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Invalid ZIP archive: {path}")
        source = load_archive(path)
        person_key = make_person_key(source)
        ccs = pd.to_numeric(source.CCCODEX, errors="coerce")
        selected = pd.DataFrame({"year": year, "person_key": person_key, "ccs": ccs})
        selected = selected.loc[selected.ccs.ge(1) & selected.ccs.le(999)].copy()
        selected["ccs"] = selected.ccs.astype(int)
        selected = selected.loc[
            [key in allowed for key in zip(selected.year, selected.person_key)]
        ].drop_duplicates(["year", "person_key", "ccs"])
        print(f"[{year}] matching condition groups={len(selected):,}; people={selected.person_key.nunique():,}", flush=True)
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    if result.empty:
        raise ValueError("No MEPS conditions matched annual diabetes records; inspect person_key format")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result):,} person-year CCS records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
