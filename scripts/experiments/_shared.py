"""Compatibility imports from the reusable project package."""

import sys

from _paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_project.metrics import best_f1_threshold as choose_f1_threshold
from diabetes_project.metrics import binary_metrics as evaluate
from diabetes_project.plots import save_diagnostic_curves as draw_curves

__all__ = ["choose_f1_threshold", "evaluate", "draw_curves"]

