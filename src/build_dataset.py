"""Helpers to build feature datasets for model training.

This module merges topology, geographic, and entropy features into a single
training table and provides lightweight validation and persistence helpers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

FeatureFormat = Literal["csv", "parquet"]


def build_feature_dataset(
    pairs: pd.DataFrame,
    topology_df: pd.DataFrame,
    geographic_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Merge feature tables on `u`, `v` and return feature matrix plus labels."""
    if not {"u", "v"}.issubset(pairs.columns):
        raise ValueError("pairs must contain 'u' and 'v' columns")

    dataset = pairs[["u", "v"]].copy()

    for frame in (topology_df, geographic_df, entropy_df):
        missing = {"u", "v"}.difference(frame.columns)
        if missing:
            raise ValueError(f"feature frame is missing required columns: {sorted(missing)}")

    dataset = dataset.merge(topology_df, on=["u", "v"], how="left")
    dataset = dataset.merge(geographic_df, on=["u", "v"], how="left")
    dataset = dataset.merge(entropy_df, on=["u", "v"], how="left")

    if isinstance(labels, pd.Series):
        labels_frame = labels.rename("label").reset_index()
        if not {"u", "v", "label"}.issubset(labels_frame.columns):
            raise ValueError(
                "labels series must be convertible to a DataFrame with 'u', 'v', and 'label' columns"
            )
        labels_frame = labels_frame[["u", "v", "label"]]
    else:
        if not {"u", "v", "label"}.issubset(labels.columns):
            raise ValueError("labels must contain 'u', 'v', and 'label' columns")
        labels_frame = labels[["u", "v", "label"]].copy()

    dataset = dataset.merge(labels_frame, on=["u", "v"], how="left")

    ordered_columns = [
        "u",
        "v",
        "common_neighbors",
        "jaccard",
        "preferential_attachment",
        "adamic_adar",
        "center_distance",
        "shared_locations",
        "location_jaccard",
        "entropy_similarity",
        "label",
    ]
    for column in ordered_columns:
        if column not in dataset.columns:
            dataset[column] = pd.NA
    dataset = dataset[ordered_columns]

    validate_feature_matrix(dataset)

    y = dataset["label"].copy()
    X = dataset.drop(columns=["label"]).copy()
    logger.info("Built feature dataset with %d rows and %d columns", len(X), len(X.columns))
    return X, y


def validate_feature_matrix(dataset: pd.DataFrame) -> None:
    """Check for missing values and duplicate `(u, v)` pairs."""
    required = {
        "u",
        "v",
        "common_neighbors",
        "jaccard",
        "preferential_attachment",
        "adamic_adar",
        "center_distance",
        "shared_locations",
        "location_jaccard",
        "entropy_similarity",
        "label",
    }
    missing_columns = required.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"dataset is missing required columns: {sorted(missing_columns)}")

    if dataset.isna().any().any():
        raise ValueError("feature matrix contains missing values")

    duplicate_pairs = dataset.duplicated(subset=["u", "v"]).sum()
    if duplicate_pairs:
        raise ValueError(f"feature matrix contains {int(duplicate_pairs)} duplicate (u, v) pairs")

    logger.info("Validated feature matrix with %d rows", len(dataset))


def save_dataset(
    dataset: pd.DataFrame,
    path: str | Path,
    format: FeatureFormat | None = None,
    index: bool = False,
) -> None:
    """Save a dataset to CSV or Parquet."""
    output_path = Path(path)
    save_format = format or output_path.suffix.lower().lstrip(".")

    if save_format not in {"csv", "parquet"}:
        raise ValueError("format must be 'csv' or 'parquet'")

    if save_format == "csv":
        dataset.to_csv(output_path, index=index)
    else:
        dataset.to_parquet(output_path, index=index)

    logger.info("Saved dataset to %s", output_path)


def build_from_raw(edges_df: pd.DataFrame, checkins_df: pd.DataFrame):
    """Backward-compatible placeholder for older callers.

    The project now builds feature tables through the modular feature pipeline,
    so this wrapper intentionally remains minimal.
    """
    raise NotImplementedError("Use build_feature_dataset() with precomputed feature frames instead.")
