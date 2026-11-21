"""Harmonize MEPS inpatient events and UCI discharges for 30-day transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from diabetes_project.harmonized_meps import RACES, hospital_bridge


def _event_keys(frame):
    out = frame.copy()
    out["person_key"] = out.person_key.astype(str)
    out["event_id"] = out.event_id.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out


def load_events(event_path, ccs_path, top_groups=64):
    events = _event_keys(pd.read_csv(event_path, dtype={"person_key": str, "event_id": str}))
    linked = _event_keys(pd.read_csv(ccs_path, dtype={"person_key": str, "event_id": str}))
    keys = ["year", "person_key", "event_id"]
    if events.duplicated(keys).any():
        raise ValueError("MEPS event identifiers are not unique")
    events = events.loc[events.readmission_30d.isin([0, 1])].copy().reset_index(drop=True)
    linked = linked.merge(events[keys], on=keys, how="inner", validate="many_to_one")
    linked["ccs"] = "CCS_" + pd.to_numeric(linked.ccs, errors="raise").astype(int).astype(str)
    linked = linked.drop_duplicates(keys + ["ccs"])
    groups = linked.ccs.value_counts().head(top_groups).index.tolist()
    return events, linked, groups


def common_matrix_uci(frame, groups):
    bridge = hospital_bridge(frame)
    out = bridge[["age", "male"] + [f"race_{name}" for name in RACES]].copy()
    out["length_of_stay"] = pd.to_numeric(frame.time_in_hospital, errors="coerce").fillna(0).clip(0, 30) / 30
    diag = frame[[f"ccs_diag_{i}" for i in (1, 2, 3)]].astype(str)
    for group in groups:
        out[f"group_{group}"] = diag.eq(group).any(axis=1).to_numpy(dtype=np.float32)
    return out.reset_index(drop=True).astype(np.float32)


def common_matrix_meps(events, linked, groups):
    out = pd.DataFrame(index=events.index)
    out["age"] = pd.to_numeric(events.age, errors="coerce").fillna(65).clip(0, 100) / 100
    out["male"] = pd.to_numeric(events.sex, errors="coerce").map({1: 1., 2: 0.}).fillna(.5)
    for name in RACES:
        out[f"race_{name}"] = events.race_common.eq(name).to_numpy(dtype=np.float32)
    out["length_of_stay"] = pd.to_numeric(events.length_of_stay, errors="coerce").fillna(0).clip(0, 30) / 30
    keys = ["year", "person_key", "event_id"]
    index = events.reset_index()[keys + ["index"]]
    match = linked.merge(index, on=keys, how="inner", validate="many_to_one")
    present = match.groupby("ccs")["index"].apply(lambda x: x.to_numpy(dtype=int)).to_dict()
    for group in groups:
        column = np.zeros(len(events), dtype=np.float32)
        if group in present:
            column[present[group]] = 1.
        out[f"group_{group}"] = column
    extra = pd.DataFrame(index=events.index)
    for col in ("emergency_admission", "any_operation", "discharge_medicine"):
        extra[col] = pd.to_numeric(events[col], errors="coerce").fillna(.5).clip(0, 1)
    return out.astype(np.float32), extra.astype(np.float32)


def group_support(uci, meps, groups):
    count = meps[[f"group_{g}" for g in groups]].sum(axis=0).to_numpy(dtype=float)
    u = uci[[f"group_{g}" for g in groups]].to_numpy(dtype=float)
    result = np.log1p(u @ count)
    return result.astype(np.float32)


def age_sex_race_agreement(uci, meps):
    """Basic diagnostics, never used as UCI target features."""
    columns = ["age", "male"] + [f"race_{name}" for name in RACES]
    return {name: {"uci_mean": float(uci[name].mean()), "meps_mean": float(meps[name].mean())}
            for name in columns}
