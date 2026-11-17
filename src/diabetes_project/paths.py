"""Canonical project paths; callers may still override every path on the CLI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

UCI_RAW = RAW_DIR / "uci" / "diabetic_data.csv"
CDC_RAW = RAW_DIR / "cdc" / "diabetes_012_health_indicators_BRFSS2015.csv"
IDS_MAPPING = RAW_DIR / "uci" / "IDS_mapping.csv"
DEFAULT_FOLDS = PROCESSED_DIR / "uci_5fold"

