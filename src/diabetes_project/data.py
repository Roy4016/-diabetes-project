"""Dataset loading and cohort definitions."""

from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDED_DISCHARGE_IDS = frozenset({11, 13, 14, 19, 20, 21})
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")


def load_uci_diabetes(path: str | Path) -> pd.DataFrame:
    """Load the UCI hospital data and create the 30-day readmission target.

    Hospice and expired encounters are excluded. Diagnosis codes remain strings,
    and patient identifiers are retained for grouped splitting and auditing only.
    """
    frame = pd.read_csv(
        path,
        na_values=["?"],
        low_memory=False,
        dtype={column: "string" for column in DIAGNOSIS_COLUMNS},
    )
    frame.columns = frame.columns.str.strip()
    required = {
        "encounter_id", "patient_nbr", "readmitted",
        "discharge_disposition_id", *DIAGNOSIS_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"UCI dataset is missing columns: {sorted(missing)}")

    discharge = pd.to_numeric(frame["discharge_disposition_id"], errors="coerce")
    frame = frame.loc[~discharge.isin(EXCLUDED_DISCHARGE_IDS)].copy()
    frame = frame.loc[frame["readmitted"].isin(["<30", ">30", "NO"])].copy()
    if "gender" in frame:
        frame = frame.loc[~frame["gender"].eq("Unknown/Invalid")].copy()
    frame["target"] = frame["readmitted"].eq("<30").astype("int8")
    frame.reset_index(drop=True, inplace=True)
    return frame


def feature_columns(frame: pd.DataFrame, extra_excluded=()) -> list[str]:
    """Return model inputs while excluding identifiers and leakage-prone fields."""
    excluded = {
        "encounter_id", "patient_nbr", "readmitted", "target",
        "discharge_disposition_id", "weight", "examide", "citoglipton",
        *extra_excluded,
    }
    return [column for column in frame.columns if column not in excluded]


def split_xy(frame: pd.DataFrame, columns: list[str]):
    return frame.loc[:, columns].copy(), frame["target"].to_numpy(dtype=np.int8)

