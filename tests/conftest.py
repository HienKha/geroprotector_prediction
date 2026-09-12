from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def balanced_labels() -> pd.Series:
    ids = [f"cmp::{index:02d}" for index in range(8)]
    return pd.Series(
        [0, 1, 0, 1, 1, 0, 1, 0],
        index=ids,
        name="label",
        dtype=int,
    )


@pytest.fixture
def unique_groups(balanced_labels: pd.Series) -> pd.Series:
    return pd.Series(
        [f"component::{index:02d}" for index in range(len(balanced_labels))],
        index=balanced_labels.index,
        name="component_id",
        dtype="string",
    )


@pytest.fixture
def two_fold_selection(balanced_labels: pd.Series):
    ids = tuple(balanced_labels.index)
    validation_a = (ids[0], ids[1], ids[4], ids[5])
    validation_b = tuple(value for value in ids if value not in validation_a)
    return [
        (validation_b, validation_a),
        (validation_a, validation_b),
    ]
