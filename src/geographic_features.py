"""Geographic / spatial feature calculations for Gowalla.

The functions in this module precompute compact lookup structures where useful
and keep pairwise feature extraction efficient for large candidate batches.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_activity_centers(checkins: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user activity centers using median latitude and longitude."""
    required = {"user", "lat", "lon"}
    missing = required.difference(checkins.columns)
    if missing:
        raise ValueError(f"checkins is missing required columns: {sorted(missing)}")

    centers = (
        checkins.groupby("user", sort=False)[["lat", "lon"]]
        .median()
        .reset_index()
        .rename(columns={"lat": "center_lat", "lon": "center_lon"})
    )

    logger.info("Computed activity centers for %d users", len(centers))
    return centers


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometers between two points."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def shared_location_count(user_to_locations: dict[int, set[int]], u: int, v: int) -> int:
    """Return the number of locations shared by users `u` and `v`."""
    locations_u = user_to_locations.get(u)
    locations_v = user_to_locations.get(v)
    if not locations_u or not locations_v:
        return 0
    return len(locations_u & locations_v)


def location_jaccard(user_to_locations: dict[int, set[int]], u: int, v: int) -> float:
    """Return the Jaccard similarity between two users' visited locations."""
    locations_u = user_to_locations.get(u)
    locations_v = user_to_locations.get(v)
    if not locations_u or not locations_v:
        return 0.0

    union = locations_u | locations_v
    if not union:
        return 0.0
    return len(locations_u & locations_v) / len(union)


def compute_geographic_features(
    pairs: Iterable[tuple[int, int]],
    user_to_locations: dict[int, set[int]],
    activity_centers: pd.DataFrame,
) -> pd.DataFrame:
    """Compute geographic features for candidate pairs.

    The activity-center lookup is built once and reused across all pairs so the
    batch stays efficient for large candidate lists.
    """
    required = {"user", "center_lat", "center_lon"}
    missing = required.difference(activity_centers.columns)
    if missing:
        raise ValueError(f"activity_centers is missing required columns: {sorted(missing)}")

    center_lookup = {
        int(row.user): (float(row.center_lat), float(row.center_lon))
        for row in activity_centers.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for u, v in pairs:
        center_u = center_lookup.get(u)
        center_v = center_lookup.get(v)
        if center_u is None or center_v is None:
            center_distance = float("nan")
        else:
            center_distance = haversine_distance(center_u[0], center_u[1], center_v[0], center_v[1])

        shared_locations = shared_location_count(user_to_locations, u, v)
        location_similarity = location_jaccard(user_to_locations, u, v)

        rows.append(
            {
                "u": u,
                "v": v,
                "center_distance": center_distance,
                "shared_locations": shared_locations,
                "location_jaccard": location_similarity,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=["u", "v", "center_distance", "shared_locations", "location_jaccard"],
    )
    logger.info("Computed geographic features for %d candidate pairs", len(frame))
    return frame


def haversine(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Compatibility wrapper for the legacy two-coordinate API."""
    return haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
