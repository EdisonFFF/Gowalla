"""Entropy-style features for users and locations.

Entropy is computed once per location and then reused to score candidate pairs
efficiently during batch feature extraction.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_location_entropy(checkins: pd.DataFrame) -> dict[int, float]:
    """Compute Shannon entropy for each location from user visit counts.

    H(l) = -Σ p(u|l) log2 p(u|l)
    """
    required = {"user", "location"}
    missing = required.difference(checkins.columns)
    if missing:
        raise ValueError(f"checkins is missing required columns: {sorted(missing)}")

    location_entropy: dict[int, float] = {}
    grouped = checkins.groupby("location", sort=False)["user"]

    # Compute once per location so the same entropy values can be reused across
    # many candidate pairs during batch feature extraction.
    for location, users in grouped:
        counts = users.value_counts(sort=False)
        location_entropy[int(location)] = shannon_entropy(counts.to_numpy())

    logger.info("Computed location entropy for %d locations", len(location_entropy))
    return location_entropy


def entropy_weighted_similarity(
    u: int,
    v: int,
    user_to_locations: dict[int, set[int]],
    location_entropy: dict[int, float],
) -> float:
    """Return entropy-weighted similarity over shared locations."""
    locations_u = user_to_locations.get(u)
    locations_v = user_to_locations.get(v)
    if not locations_u or not locations_v:
        return 0.0

    shared_locations = locations_u & locations_v
    if not shared_locations:
        return 0.0

    # Weight overlap by inverse location entropy so more informative locations
    # contribute more strongly than very common ones.
    return float(
        sum(1.0 / (1.0 + location_entropy.get(location, 0.0)) for location in shared_locations)
    )


def compute_entropy_features(
    pairs: Iterable[tuple[int, int]],
    user_to_locations: dict[int, set[int]],
    location_entropy: dict[int, float],
) -> pd.DataFrame:
    """Compute entropy-based similarity features for candidate pairs."""
    rows: list[dict[str, Any]] = []

    for u, v in pairs:
        rows.append(
            {
                "u": u,
                "v": v,
                "entropy_similarity": entropy_weighted_similarity(
                    u,
                    v,
                    user_to_locations,
                    location_entropy,
                ),
            }
        )

    frame = pd.DataFrame(rows, columns=["u", "v", "entropy_similarity"])
    logger.info("Computed entropy features for %d candidate pairs", len(frame))
    return frame


def shannon_entropy(counts: Iterable[int]) -> float:
    """Backward-compatible entropy helper for generic count sequences."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return float(entropy)
