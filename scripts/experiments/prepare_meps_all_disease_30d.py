#!/usr/bin/env python3
"""Derive 30-day readmission labels for every MEPS inpatient event, 2008-2012.

No diabetes restriction or annual person table is applied. This uses the same
exact-date, uncertainty, and follow-up rules as the diabetic event builder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_project.meps_events import (  # noqa: E402
    ANNUAL_FEATURES, EVENT_PUF, _date, _month_interval_intersects, _next_after, _numeric,
    _person_key, _read_archive,
)


def build(event_root: Path, annual_path: Path, outdir: Path):
    inpatient_parts, er_parts = [], []
    for year, puf in EVENT_PUF.items():
        inpatient = _read_archive(event_root / f"h{puf}dssp.zip")
        inpatient_parts.append(pd.DataFrame({
            "year": year, "person_key": _person_key(inpatient),
            "event_id": inpatient.EVNTIDX.astype("string"),
            "admission_date": _date(inpatient, "IPBEG"),
            "discharge_date": _date(inpatient, "IPEND"),
            "admission_year": _numeric(inpatient.IPBEGYR),
            "admission_month": _numeric(inpatient.IPBEGMM),
            "length_of_stay": _numeric(inpatient.NUMNIGHX),
            "emergency_admission": _numeric(inpatient.EMERROOM).map({1.: 1., 2.: 0.}),
            "any_operation": _numeric(inpatient.ANYOPER).map({1.: 1., 2.: 0.}),
            "discharge_medicine": _numeric(inpatient.DSCHPMED).map({1.: 1., 2.: 0.}),
            "reason_for_stay": _numeric(inpatient.RSNINHOS),
        }))
        er = _read_archive(event_root / f"h{puf}essp.zip")
        er_parts.append(pd.DataFrame({
            "person_key": _person_key(er), "er_date": _date(er, "ERDATE"),
            "er_year": _numeric(er.ERDATEYR), "er_month": _numeric(er.ERDATEMM),
        }))
    inpatient = pd.concat(inpatient_parts, ignore_index=True)
    er = pd.concat(er_parts, ignore_index=True)
    index = inpatient.loc[inpatient.admission_date.notna() & inpatient.discharge_date.notna()].copy()
    index = index.loc[index.discharge_date.ge(index.admission_date)].reset_index(drop=True)
    followup_stop = index.discharge_date + pd.Timedelta(days=30)
    next_ip = _next_after(inpatient.person_key.to_numpy(), inpatient.admission_date.to_numpy(),
                          index.person_key.to_numpy(), index.discharge_date.to_numpy())
    next_er = _next_after(er.person_key.to_numpy(), er.er_date.to_numpy(),
                          index.person_key.to_numpy(), index.discharge_date.to_numpy())
    index["days_to_next_inpatient"] = (next_ip - index.discharge_date).dt.days
    index["days_to_next_er"] = (next_er - index.discharge_date).dt.days
    index["readmission_30d"] = index.days_to_next_inpatient.between(1, 30).astype(float)
    index["er_revisit_30d"] = index.days_to_next_er.between(1, 30).astype(float)

    unknown_ip = inpatient.loc[inpatient.admission_date.isna()]
    unknown_er = er.loc[er.er_date.isna()]
    uncertain_ip = np.zeros(len(index), dtype=bool)
    uncertain_er = np.zeros(len(index), dtype=bool)
    for person, positions in index.groupby("person_key").groups.items():
        pos = np.asarray(list(positions), dtype=int)
        ip_unknown = unknown_ip.loc[unknown_ip.person_key.eq(person)]
        er_unknown = unknown_er.loc[unknown_er.person_key.eq(person)]
        uncertain_ip[pos] = _month_interval_intersects(
            ip_unknown.admission_year.to_numpy(), ip_unknown.admission_month.to_numpy(),
            index.discharge_date.iloc[pos].to_numpy(), followup_stop.iloc[pos].to_numpy())
        uncertain_er[pos] = _month_interval_intersects(
            er_unknown.er_year.to_numpy(), er_unknown.er_month.to_numpy(),
            index.discharge_date.iloc[pos].to_numpy(), followup_stop.iloc[pos].to_numpy())
    index.loc[uncertain_ip & index.readmission_30d.eq(0), "readmission_30d"] = np.nan
    index.loc[uncertain_er & index.er_revisit_30d.eq(0), "er_revisit_30d"] = np.nan
    index.loc[followup_stop.gt(pd.Timestamp("2012-12-31")),
              ["readmission_30d", "er_revisit_30d"]] = np.nan
    events_before_annual_merge = len(index)
    annual = pd.read_csv(annual_path, dtype={"person_key": str})
    needed = ["year", "person_key", *ANNUAL_FEATURES]
    if missing := set(needed) - set(annual):
        raise ValueError(f"Full annual MEPS bridge missing {sorted(missing)}")
    index = index.merge(annual[needed], on=["year", "person_key"], how="inner", validate="many_to_one")
    if index.duplicated(["year", "person_key", "event_id"]).any():
        raise ValueError("MEPS event keys are not unique")
    outdir.mkdir(parents=True, exist_ok=True)
    index.to_csv(outdir / "meps_all_disease_event_30d_all.csv", index=False)
    modeling = index.loc[index.readmission_30d.notna()].copy()
    modeling.to_csv(outdir / "meps_all_disease_event_30d_model.csv", index=False)
    summary = {
        "source_inpatient_events": int(len(inpatient)),
        "events_with_valid_dates_before_annual_merge": int(events_before_annual_merge),
        "index_events_with_valid_dates": int(len(index)),
        "index_events_with_certain_readmission_30d": int(len(modeling)),
        "people_with_certain_readmission_30d": int(modeling.person_key.nunique()),
        "readmission_30d_positive": int(modeling.readmission_30d.sum()),
        "readmission_30d_rate": float(modeling.readmission_30d.mean()),
        "both_readmission_and_er_labels_certain": int(index[["readmission_30d", "er_revisit_30d"]].notna().all(axis=1).sum()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=ROOT / "data/raw/meps/events_2008_2024")
    parser.add_argument("--annual", type=Path, default=ROOT / "data/processed/meps/meps_all_disease_bridge8_2008_2012.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/processed/meps_events_all_disease_2008_2012")
    args = parser.parse_args()
    build(args.event_root, args.annual, args.outdir)


if __name__ == "__main__":
    main()
