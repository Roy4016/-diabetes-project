#!/usr/bin/env python3
"""Link MEPS inpatient events to their reported CCS conditions through HC event appendices."""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from _paths import PROJECT_ROOT
from prepare_meps_conditions import PUFS, load_archive


EVENT_PUFS = {2008: 118, 2009: 126, 2010: 135, 2011: 144, 2012: 152}


def normalized_id(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def read_link(path):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith((".ssp", ".xpt", ".xport"))]
        if len(members) != 1:
            raise ValueError(f"Expected one condition-event link file in {path}: {archive.namelist()}")
        frame = pd.read_sas(io.BytesIO(archive.read(members[0])), format="xport", encoding="latin-1")
    frame.columns = frame.columns.str.upper()
    missing = {"CONDIDX", "EVNTIDX", "DUPERSID"} - set(frame.columns)
    if missing:
        raise ValueError(f"Link file missing {sorted(missing)}: {path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-raw", type=Path, default=PROJECT_ROOT / "data/raw/meps/conditions_2008_2012")
    parser.add_argument("--link-raw", type=Path, default=PROJECT_ROOT / "data/raw/meps/condition_event_links_2008_2012")
    parser.add_argument("--event-data", type=Path, default=PROJECT_ROOT / "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--years", nargs="+", type=int, default=list(EVENT_PUFS), choices=EVENT_PUFS)
    args = parser.parse_args()
    args.link_raw.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
    events["event_id"] = normalized_id(events.event_id)
    if events.duplicated(["year", "person_key", "event_id"]).any():
        raise ValueError("Event dataset contains duplicate year/person/event keys")
    keys = events[["year", "person_key", "event_id"]].drop_duplicates()
    rows = []
    for year in args.years:
        puf = EVENT_PUFS[year]
        path = args.link_raw / f"h{puf}if1ssp.zip"
        if not path.exists():
            url = f"https://meps.ahrq.gov/mepsweb/data_files/pufs/h{puf}if1ssp.zip"
            print(f"[{year}] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path)
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Invalid ZIP archive: {path}")
        condition_path = args.conditions_raw / f"h{PUFS[year]}ssp.zip"
        if not condition_path.exists():
            raise FileNotFoundError(f"Run prepare_meps_conditions.py first: {condition_path}")
        condition = load_archive(condition_path)
        if "CONDIDX" not in condition or "ICD9CODX" not in condition:
            raise ValueError(f"Condition PUF needs CONDIDX and ICD9CODX: {condition_path}")
        condition = pd.DataFrame({
            "cond_id": normalized_id(condition.CONDIDX),
            "person_id": normalized_id(condition.DUPERSID),
            "ccs": pd.to_numeric(condition.CCCODEX, errors="coerce"),
            "icd9_prefix": normalized_id(condition.ICD9CODX).str.upper().str.replace(".", "", regex=False),
        })
        condition = condition[condition.ccs.between(1, 999)].copy()
        condition["ccs"] = condition.ccs.astype(int)
        if condition.duplicated(["cond_id", "person_id"]).any():
            raise ValueError(f"Duplicate condition identifiers in {condition_path}")
        link = read_link(path)
        link = pd.DataFrame({
            "cond_id": normalized_id(link.CONDIDX),
            "person_id": normalized_id(link.DUPERSID),
            "event_id": normalized_id(link.EVNTIDX),
        })
        linked = link.merge(condition, on=["cond_id", "person_id"], how="inner", validate="many_to_one")
        linked["year"] = year
        # Match person ID as well as event ID; a condition linked to a later
        # visit cannot be assigned to an earlier inpatient stay.
        year_events = keys[keys.year.eq(year)].copy()
        year_events["person_id"] = year_events.person_key.str.split("_", n=1).str[-1]
        linked = year_events.merge(
            linked[["year", "person_id", "event_id", "ccs", "icd9_prefix"]],
            on=["year", "person_id", "event_id"], how="inner",
        )[["year", "person_key", "event_id", "ccs", "icd9_prefix"]].drop_duplicates()
        print(f"[{year}] linked events={linked.event_id.nunique():,} CCS records={len(linked):,}", flush=True)
        rows.append(linked)
    result = pd.concat(rows, ignore_index=True)
    if result.empty:
        raise ValueError("No inpatient events linked to CCS conditions; inspect ID formats")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result):,} event-CCS records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
