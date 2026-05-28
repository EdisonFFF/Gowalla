"""Candidate generation for the Gowalla link prediction pipeline.

The module provides scalable neighborhood-based and location-based candidate
construction utilities for downstream topology and geographic feature
extraction.
"""
from __future__ import annotations

import itertools
import logging
from statistics import median
from collections.abc import Iterable

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

Edge = tuple[int, int]

_LAST_2HOP_CANDIDATES: dict[int, set[int]] | None = None


def _canonical_edge(u: int, v: int) -> Edge:
    """Return an undirected edge in canonical order."""
    return (u, v) if u <= v else (v, u)


def generate_2hop_candidates(
    G: nx.Graph,
    users: Iterable[int] | None = None,
) -> dict[int, set[int]]:
    """Generate 2-hop friendship candidates for each user.

    For each user u, the candidate set is the union of neighbors of neighbors,
    excluding direct neighbors and the user itself.
    """
    global _LAST_2HOP_CANDIDATES

    selected_users = G.nodes() if users is None else users
    candidates: dict[int, set[int]] = {}

    total_candidates = 0
    max_candidates = 0

    for user in selected_users:
        direct_neighbors = set(G.neighbors(user))
        two_hop_neighbors: set[int] = set()
        for neighbor in direct_neighbors:
            two_hop_neighbors.update(G.neighbors(neighbor))

        two_hop_neighbors.difference_update(direct_neighbors)
        two_hop_neighbors.discard(user)

        candidates[user] = two_hop_neighbors
        count = len(two_hop_neighbors)
        total_candidates += count
        max_candidates = max(max_candidates, count)

    _LAST_2HOP_CANDIDATES = candidates

    avg_candidates = total_candidates / len(candidates) if candidates else 0.0
    logger.info(
        "Generated 2-hop candidates for %d users | total=%d avg=%.2f max=%d",
        len(candidates),
        total_candidates,
        avg_candidates,
        max_candidates,
    )
    return candidates


def generate_place_friend_candidates(
    user_to_locations: dict[int, set[int]],
    location_to_users: dict[int, set[int]],
) -> set[Edge]:
    """Generate candidate friendship pairs for users sharing at least one location."""
    candidates: set[Edge] = set()

    for users in location_to_users.values():
        if len(users) < 2:
            continue
        for u, v in itertools.combinations(sorted(users), 2):
            candidates.add((u, v))

    logger.info(
        "Generated %d place-friend candidates from %d locations and %d users",
        len(candidates),
        len(location_to_users),
        len(user_to_locations),
    )
    return candidates


def sample_negative_edges(
    G: nx.Graph,
    num_samples: int,
    strategy: str = "random",
    random_state: int = 42,
) -> list[Edge]:
    """Sample negative edges using random or 2-hop strategies."""
    if num_samples <= 0:
        return []

    rng = np.random.default_rng(random_state)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []

    existing_edges = {_canonical_edge(u, v) for u, v in G.edges() if u != v}

    strategy_key = strategy.lower().replace("-", "").replace("_", "")

    if strategy_key == "random":
        samples: list[Edge] = []
        sampled_edges: set[Edge] = set()
        while len(samples) < num_samples:
            u, v = rng.choice(nodes, size=2, replace=False)
            edge = _canonical_edge(int(u), int(v))
            if edge in existing_edges or edge in sampled_edges:
                continue
            sampled_edges.add(edge)
            samples.append(edge)

        logger.info("Sampled %d random negative edges", len(samples))
        return samples

    if strategy_key in {"2hop", "twohop"}:
        candidates = generate_2hop_candidates(G)
        two_hop_edges = {
            _canonical_edge(u, v)
            for u, neighbor_set in candidates.items()
            for v in neighbor_set
            if u != v
        }
        negative_candidates = sorted(two_hop_edges - existing_edges)
        if not negative_candidates:
            logger.info("No 2-hop negative candidates available")
            return []

        replace = len(negative_candidates) < num_samples
        indices = rng.choice(len(negative_candidates), size=num_samples if replace else min(num_samples, len(negative_candidates)), replace=replace)
        samples = [negative_candidates[int(index)] for index in np.atleast_1d(indices)]

        logger.info("Sampled %d 2-hop negative edges", len(samples))
        return samples

    raise ValueError("strategy must be 'random' or '2hop'")


def estimate_candidate_statistics() -> dict[str, float | int]:
    """Estimate summary statistics for the most recent 2-hop candidate set."""
    if _LAST_2HOP_CANDIDATES is None:
        raise ValueError("No 2-hop candidates have been generated yet.")

    counts = [len(candidate_set) for candidate_set in _LAST_2HOP_CANDIDATES.values()]
    total_candidates = int(sum(counts))
    avg_candidates = float(total_candidates / len(counts)) if counts else 0.0
    max_candidates = int(max(counts)) if counts else 0
    median_candidates = float(median(counts)) if counts else 0.0

    stats = {
        "total_candidates": total_candidates,
        "avg_candidates_per_user": avg_candidates,
        "max_candidates": max_candidates,
        "median_candidates": median_candidates,
    }
    logger.info("Candidate statistics: %s", stats)
    return stats


def generate_candidates(graph: nx.Graph, k_hop: int = 2) -> list[Edge]:
    """Backward-compatible helper returning unique pairs up to `k_hop` distance."""
    if k_hop < 1:
        return []

    if k_hop == 2:
        candidate_map = generate_2hop_candidates(graph)
        return sorted(
            {
                _canonical_edge(u, v)
                for u, neighbor_set in candidate_map.items()
                for v in neighbor_set
                if u != v
            }
        )

    pairs: set[Edge] = set()
    for node in graph.nodes():
        for neighbor in nx.single_source_shortest_path_length(graph, node, cutoff=k_hop):
            if node >= neighbor:
                continue
            pairs.add((node, neighbor))
    return sorted(pairs)
