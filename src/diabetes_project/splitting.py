"""Patient-disjoint holdout and cross-validation splitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


@dataclass(frozen=True)
class PatientFold:
    number: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_train_idx: np.ndarray
    val_idx: np.ndarray


def validate_disjoint(*frames: pd.DataFrame, group_column: str = "patient_nbr") -> None:
    groups = [set(frame[group_column].dropna()) for frame in frames]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            overlap = groups[left].intersection(groups[right])
            if overlap:
                raise ValueError(
                    f"Patient leakage between partitions {left} and {right}: "
                    f"{len(overlap)} overlapping groups"
                )


def make_patient_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 777,
    group_column: str = "patient_nbr",
    target_column: str = "target",
    validation_size: float = 0.20,
) -> list[PatientFold]:
    """Create stratified outer folds and grouped inner validation partitions."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for number, (train_idx, test_idx) in enumerate(
        outer.split(frame, frame[target_column], groups=frame[group_column]), start=1
    ):
        outer_train = frame.iloc[train_idx]
        inner = GroupShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=seed + number
        )
        inner_local, val_local = next(
            inner.split(outer_train, groups=outer_train[group_column])
        )
        inner_idx = train_idx[inner_local]
        val_idx = train_idx[val_local]
        validate_disjoint(frame.iloc[inner_idx], frame.iloc[val_idx], frame.iloc[test_idx],
                          group_column=group_column)
        folds.append(PatientFold(number, train_idx, test_idx, inner_idx, val_idx))
    return folds

