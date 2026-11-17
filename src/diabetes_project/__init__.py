"""Shared code for the diabetes readmission project."""

from .data import load_uci_diabetes
from .splitting import make_patient_folds, validate_disjoint

__all__ = ["load_uci_diabetes", "make_patient_folds", "validate_disjoint"]

