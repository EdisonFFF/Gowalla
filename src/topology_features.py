"""Topological feature computations for candidate node pairs."""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _neighbor_cache(graph: nx.Graph) -> dict[int, set[int]]:
    """Build a cached neighbor-set lookup for repeated pair scoring."""
    return {int(node): set(graph.neighbors(node)) for node in graph.nodes()}


def common_neighbors_score(graph: nx.Graph, u: int, v: int) -> int:
    """Return the number of shared neighbors between two nodes."""
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    return len(neighbors_u & neighbors_v)


def jaccard_score(graph: nx.Graph, u: int, v: int) -> float:
    """Return the Jaccard similarity between two nodes' neighborhoods."""
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    union_size = len(neighbors_u | neighbors_v)
    if union_size == 0:
        return 0.0
    return len(neighbors_u & neighbors_v) / union_size


def preferential_attachment_score(graph: nx.Graph, u: int, v: int) -> int:
    """Return the preferential attachment score for a node pair."""
    return graph.degree(u) * graph.degree(v)


def adamic_adar_score(graph: nx.Graph, u: int, v: int) -> float:
    """Return the Adamic-Adar score for a node pair."""
    common_neighbors = set(graph.neighbors(u)) & set(graph.neighbors(v))
    score = 0.0
    for neighbor in common_neighbors:
        degree = graph.degree(neighbor)
        if degree > 1:
            score += 1.0 / math.log(degree)
    return score


def compute_topology_features(
    graph: nx.Graph,
    pairs: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    """Compute topology features for an iterable of candidate pairs.

    The function caches neighborhood sets once so large candidate batches can be
    scored without repeatedly recomputing graph adjacency information.
    """
    neighbor_cache = _neighbor_cache(graph)
    rows: list[dict[str, Any]] = []

    for u, v in pairs:
        neighbors_u = neighbor_cache.get(u)
        neighbors_v = neighbor_cache.get(v)
        if neighbors_u is None or neighbors_v is None:
            common_neighbors = 0
            jaccard = 0.0
            preferential_attachment = 0
            adamic_adar = 0.0
        else:
            common = neighbors_u & neighbors_v
            union_size = len(neighbors_u | neighbors_v)
            common_neighbors = len(common)
            jaccard = (common_neighbors / union_size) if union_size else 0.0
            preferential_attachment = len(neighbors_u) * len(neighbors_v)
            adamic_adar = sum(
                1.0 / math.log(len(neighbor_cache[neighbor]))
                for neighbor in common
                if len(neighbor_cache[neighbor]) > 1
            )

        rows.append(
            {
                "u": u,
                "v": v,
                "common_neighbors": common_neighbors,
                "jaccard": jaccard,
                "preferential_attachment": preferential_attachment,
                "adamic_adar": adamic_adar,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "u",
            "v",
            "common_neighbors",
            "jaccard",
            "preferential_attachment",
            "adamic_adar",
        ],
    )

    logger.info("Computed topology features for %d candidate pairs", len(frame))
    return frame


def common_neighbors_count(graph: nx.Graph, u: int, v: int) -> int:
    """Backward-compatible wrapper for common-neighbor counting."""
    return common_neighbors_score(graph, u, v)


def jaccard_coefficient(graph: nx.Graph, u: int, v: int) -> float:
    """Backward-compatible wrapper for Jaccard similarity."""
    return jaccard_score(graph, u, v)
