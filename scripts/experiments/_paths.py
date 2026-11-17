"""Stable defaults for historical experiment entry points."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UCI_RAW = str(PROJECT_ROOT / "data/raw/uci/diabetic_data.csv")
CDC_RAW = str(PROJECT_ROOT / "data/raw/cdc/diabetes_012_health_indicators_BRFSS2015.csv")
CCS_RAW = str(PROJECT_ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
UCI_HOLDOUT = str(PROJECT_ROOT / "data/processed/uci_holdout_with_ids")
MEPS_BRIDGE = str(PROJECT_ROOT / "data/processed/meps/meps_diabetes_bridge8.csv")


def result_dir(name: str) -> str:
    return str(PROJECT_ROOT / "results" / name)

