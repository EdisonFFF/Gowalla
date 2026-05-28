"""Helpers to build feature datasets for model training.

Functions:
- build_from_raw(edges_df, checkins_df) -> (X, y)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_from_raw(edges_df: pd.DataFrame, checkins_df: pd.DataFrame):
    """Placeholder: build features X and labels y from raw tables.

    Current implementation returns empty arrays; fill in feature logic.
    """
    X = np.zeros((0, 0))
    y = np.zeros((0,))
    return X, y
