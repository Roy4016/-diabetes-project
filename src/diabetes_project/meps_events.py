"""Build exact 30-day MEPS inpatient readmission and ER revisit labels."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_PUF = {2008: 118, 2009: 126, 2010: 135, 2011: 144, 2012: 152}
ANNUAL_FEATURES = [
    "age", "sex", "high_bp", "high_chol", "stroke", "heart_disease", "mi",
    "race_common", "physical_health", "mental_health", "general_health",
    "activity_limitation", "er_visits", "inpatient_nights", "total_expense",
]


def _read_archive(path):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist()
                   if name.lower().endswith((".ssp", ".xpt", ".dta"))]
        if len(members) != 1:
            raise ValueError(f"Unexpected event archive members in {path}: {archive.namelist()}")
        raw = archive.read(members[0])
    if members[0].lower().endswith(".dta"):
        frame = pd.read_stata(io.BytesIO(raw), convert_categoricals=False)
    else:
        frame = pd.read_sas(io.BytesIO(raw), format="xport", encoding="latin-1")
    frame.columns = frame.columns.str.upper()
    return frame


def _numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values.abs() < 1e-50, 0.)
    return values.mask(values < 0)


def _person_key(frame):
    panel = _numeric(frame["PANEL"]).astype("Int64").astype("string")
    identifier = frame["DUPERSID"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    redundant = identifier.str.len().eq(10) & identifier.str[:2].eq(panel)
    identifier = identifier.where(~redundant, identifier.str[2:])
    return panel + "_" + identifier


def _date(frame, prefix):
    values = {
        "year": _numeric(frame[f"{prefix}YR"]),
        "month": _numeric(frame[f"{prefix}MM"]),
        "day": _numeric(frame[f"{prefix}DD"]),
    }
    return pd.to_datetime(pd.DataFrame(values), errors="coerce")


def _month_interval_intersects(year, month, starts, stops):
    """Mark unknown-day event months that overlap each open/closed follow-up interval."""
    result = np.zeros(len(starts), dtype=bool)
    valid = np.isfinite(year) & np.isfinite(month)
    if not valid.any():
        return result
    month_start = pd.to_datetime(pd.DataFrame({
        "year": year[valid].astype(int), "month": month[valid].astype(int), "day": 1,
    }), errors="coerce")
    month_stop = month_start + pd.offsets.MonthEnd(1)
    for left, right in zip(month_start.dropna(), month_stop[month_start.notna()]):
        result |= (right > starts) & (left <= stops)
    return result


def _next_after(persons, event_dates, index_persons, index_stops):
    by_person = {}
    for person, date in zip(persons, event_dates):
        if pd.notna(date):
            by_person.setdefault(person, []).append(np.datetime64(date, "ns"))
    for values in by_person.values():
        values.sort()
    output = np.full(len(index_stops), np.datetime64("NaT"), dtype="datetime64[ns]")
    for position, (person, stop) in enumerate(zip(index_persons, index_stops)):
        values = by_person.get(person)
        if values is None or pd.isna(stop):
            continue
        array = np.asarray(values, dtype="datetime64[ns]")
        location = np.searchsorted(array, np.datetime64(stop, "ns"), side="right")
        if location < len(array):
            output[position] = array[location]
    return pd.to_datetime(output)


def build_event_dataset(event_root, annual_path, output_dir):
    event_root, output = Path(event_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inpatient_parts, er_parts = [], []
    for year, puf in EVENT_PUF.items():
        inpatient = _read_archive(event_root / f"h{puf}dssp.zip")
        inpatient_parts.append(pd.DataFrame({
            "year": year, "person_key": _person_key(inpatient),
            "event_id": inpatient["EVNTIDX"].astype("string"),
            "admission_date": _date(inpatient, "IPBEG"),
            "discharge_date": _date(inpatient, "IPEND"),
            "admission_year": _numeric(inpatient["IPBEGYR"]),
            "admission_month": _numeric(inpatient["IPBEGMM"]),
            "length_of_stay": _numeric(inpatient["NUMNIGHX"]),
            "emergency_admission": _numeric(inpatient["EMERROOM"]).map({1.: 1., 2.: 0.}),
            "any_operation": _numeric(inpatient["ANYOPER"]).map({1.: 1., 2.: 0.}),
            "discharge_medicine": _numeric(inpatient["DSCHPMED"]).map({1.: 1., 2.: 0.}),
            "reason_for_stay": _numeric(inpatient["RSNINHOS"]),
        }))
        er = _read_archive(event_root / f"h{puf}essp.zip")
        er_parts.append(pd.DataFrame({
            "year": year, "person_key": _person_key(er),
            "er_date": _date(er, "ERDATE"),
            "er_year": _numeric(er["ERDATEYR"]), "er_month": _numeric(er["ERDATEMM"]),
        }))

    inpatient = pd.concat(inpatient_parts, ignore_index=True)
    er = pd.concat(er_parts, ignore_index=True)
    index = inpatient[inpatient.admission_date.notna() & inpatient.discharge_date.notna()].copy()
    index = index[index.discharge_date.ge(index.admission_date)].reset_index(drop=True)
    followup_stop = index.discharge_date + pd.Timedelta(days=30)
    next_ip = _next_after(
        inpatient.person_key.to_numpy(), inpatient.admission_date.to_numpy(),
        index.person_key.to_numpy(), index.discharge_date.to_numpy(),
    )
    next_er = _next_after(
        er.person_key.to_numpy(), er.er_date.to_numpy(),
        index.person_key.to_numpy(), index.discharge_date.to_numpy(),
    )
    index["days_to_next_inpatient"] = (next_ip - index.discharge_date).dt.days
    index["days_to_next_er"] = (next_er - index.discharge_date).dt.days
    index["readmission_30d"] = index.days_to_next_inpatient.between(1, 30).astype(float)
    index["er_revisit_30d"] = index.days_to_next_er.between(1, 30).astype(float)

    unknown_ip = inpatient[inpatient.admission_date.isna()]
    unknown_er = er[er.er_date.isna()]
    uncertain_ip = np.zeros(len(index), dtype=bool); uncertain_er = np.zeros(len(index), dtype=bool)
    for person, positions in index.groupby("person_key").groups.items():
        pos = np.asarray(list(positions), dtype=int)
        ip_unknown = unknown_ip[unknown_ip.person_key.eq(person)]
        er_unknown = unknown_er[unknown_er.person_key.eq(person)]
        uncertain_ip[pos] = _month_interval_intersects(
            ip_unknown.admission_year.to_numpy(), ip_unknown.admission_month.to_numpy(),
            index.discharge_date.iloc[pos].to_numpy(), followup_stop.iloc[pos].to_numpy(),
        )
        uncertain_er[pos] = _month_interval_intersects(
            er_unknown.er_year.to_numpy(), er_unknown.er_month.to_numpy(),
            index.discharge_date.iloc[pos].to_numpy(), followup_stop.iloc[pos].to_numpy(),
        )
    index["readmission_label_certain"] = ~uncertain_ip
    index["er_label_certain"] = ~uncertain_er
    index.loc[uncertain_ip & index.readmission_30d.eq(0), "readmission_30d"] = np.nan
    index.loc[uncertain_er & index.er_revisit_30d.eq(0), "er_revisit_30d"] = np.nan
    index["followup_complete"] = followup_stop.le(pd.Timestamp("2012-12-31"))
    index.loc[~index.followup_complete, ["readmission_30d", "er_revisit_30d"]] = np.nan

    annual = pd.read_csv(annual_path, dtype={"person_key": str})
    needed = ["year", "person_key", *ANNUAL_FEATURES]
    missing = set(needed).difference(annual.columns)
    if missing:
        raise ValueError(f"Annual MEPS bridge is missing: {sorted(missing)}")
    index = index.merge(annual[needed], on=["year", "person_key"], how="inner", validate="many_to_one")
    index["composite_30d"] = np.where(
        index[["readmission_30d", "er_revisit_30d"]].notna().all(axis=1),
        index[["readmission_30d", "er_revisit_30d"]].max(axis=1), np.nan,
    )
    index.to_csv(output / "meps_diabetes_event_30d_all.csv", index=False)
    modeling = index[index.readmission_30d.notna()].copy()
    modeling.to_csv(output / "meps_diabetes_event_30d_model.csv", index=False)
    yearly = modeling.groupby("year").agg(
        IndexEvents=("event_id", "size"), UniquePeople=("person_key", "nunique"),
        Readmissions30d=("readmission_30d", "sum"),
        Readmission30dRate=("readmission_30d", "mean"),
    ).reset_index()
    yearly.to_csv(output / "year_summary.csv", index=False)
    summary = {
        "all_inpatient_events": int(len(inpatient)),
        "diabetes_index_events_after_annual_merge": int(len(index)),
        "diabetes_index_events_with_certain_readmission_label": int(len(modeling)),
        "unique_diabetes_people": int(modeling.person_key.nunique()),
        "readmission_30d_rate": float(modeling.readmission_30d.mean()),
        "readmission_30d_positive": int(modeling.readmission_30d.sum()),
        "er_revisit_observed": int(index.er_revisit_30d.notna().sum()),
        "er_revisit_30d_rate": float(index.er_revisit_30d.mean()),
        "composite_observed": int(index.composite_30d.notna().sum()),
        "composite_30d_rate": float(index.composite_30d.mean()),
        "excluded_uncertain_readmission": int(index.readmission_30d.isna().sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return index, modeling, summary
