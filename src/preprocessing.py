"""Preprocessing utilities for the Gowalla link prediction pipeline.

This module prepares the graph and check-in tables for downstream candidate
generation, topology features, geographic features, and entropy-aware feature
computation. The emphasis is on reproducible, scalable preprocessing for a
research pipeline.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Edge = tuple[int, int]


def _canonical_edge(u: int, v: int) -> Edge:
    """Return an undirected edge in canonical sorted order."""
    return (u, v) if u <= v else (v, u)


def filter_active_users(
    G: nx.Graph,
    checkins: pd.DataFrame,
    min_degree: int = 5,
    min_checkins: int = 10,
) -> tuple[nx.Graph, pd.DataFrame, set[int]]:
    """Filter inactive users from the graph and check-in table.

    Active-user filtering is important because it removes sparse nodes that add
    noise, reduce statistical power, and inflate the candidate space without
    contributing enough evidence for learning or evaluation.
    """
    if "user" not in checkins.columns:
        raise ValueError("checkins must contain a 'user' column")

    original_node_count = G.number_of_nodes()

    degree_map = dict(G.degree())
    degree_active_users = {node for node, degree in degree_map.items() if degree >= min_degree}

    checkin_counts = checkins.groupby("user", sort=False).size()
    checkin_active_users = set(checkin_counts[checkin_counts >= min_checkins].index)

    active_users = degree_active_users & checkin_active_users

    # Preserve the original connectivity structure induced by the active-user set.
    filtered_graph = G.subgraph(active_users).copy()
    filtered_checkins = checkins[checkins["user"].isin(active_users)].copy()

    logger.info("Original nodes: %d", original_node_count)
    logger.info("Filtered nodes: %d", filtered_graph.number_of_nodes())
    logger.info("Filtered edges: %d", filtered_graph.number_of_edges())

    return filtered_graph, filtered_checkins, active_users


def create_train_test_split(
    G: nx.Graph,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[nx.Graph, list[Edge], list[Edge]]:
    """Split undirected friendship edges into train and test sets.

    The split is reproducible and keeps a spanning forest in the training graph
    so that connectivity is preserved whenever the original graph is connected
    and enough non-forest edges exist for a test split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    edges = [
        _canonical_edge(u, v)
        for u, v in G.edges()
        if u != v
    ]
    edges = list(dict.fromkeys(edges))

    rng = np.random.default_rng(random_state)

    protected_edges: set[Edge] = set()
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        for u, v in nx.dfs_edges(subgraph, source=next(iter(component))):
            protected_edges.add(_canonical_edge(u, v))

    candidate_edges = [edge for edge in edges if edge not in protected_edges]
    rng.shuffle(candidate_edges)

    desired_test_size = int(round(len(edges) * test_size))
    test_count = min(desired_test_size, len(candidate_edges))

    test_edges = candidate_edges[:test_count]
    test_edge_set = set(test_edges)
    train_edges = [edge for edge in edges if edge not in test_edge_set]

    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    original_components = nx.number_connected_components(G)
    train_components = nx.number_connected_components(G_train)
    if original_components == 1 and train_components != 1:
        logger.warning("Training graph became disconnected after split.")

    return G_train, train_edges, test_edges


def build_user_location_maps(
    checkins: pd.DataFrame,
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, list[tuple[Any, ...]]]]:
    """Precompute lookup maps for feature extraction and entropy statistics.

    These lookup maps are precomputed so repeated candidate scoring does not pay
    the cost of scanning the full check-in table on every feature request.
    """
    required = ["user", "location"]
    missing = [column for column in required if column not in checkins.columns]
    if missing:
        raise ValueError(f"checkins is missing required columns: {missing}")

    grouped = checkins.groupby("user", sort=False)
    user_to_locations: dict[int, set[int]] = {
        int(user): set(group["location"].astype(int).tolist())
        for user, group in grouped
    }

    location_to_users: dict[int, set[int]] = {
        int(location): set(group["user"].astype(int).tolist())
        for location, group in checkins.groupby("location", sort=False)
    }

    # Storing per-user rows as compact tuples keeps the structure lightweight
    # while still enabling repeated lookup during feature engineering.
    user_to_checkins: dict[int, list[tuple[Any, ...]]] = {
        int(user): list(group.itertuples(index=False, name=None))
        for user, group in grouped
    }

    return user_to_locations, location_to_users, user_to_checkins


def compute_basic_graph_stats(G: nx.Graph) -> dict[str, float | int]:
    """Compute summary statistics for reporting and evaluation."""
    node_count = int(G.number_of_nodes())
    edge_count = int(G.number_of_edges())
    density = float(nx.density(G))
    average_degree = float((2.0 * edge_count / node_count) if node_count else 0.0)
    connected_components = int(nx.number_connected_components(G))

    return {
        "nodes": node_count,
        "edges": edge_count,
        "density": density,
        "average_degree": average_degree,
        "connected_components": connected_components,
    }


if __name__ == "__main__":
    from src import load_data

    data_dir = "data"
    edges_path = f"{data_dir}/loc-gowalla_edges.txt"
    checkins_path = f"{data_dir}/loc-gowalla_totalCheckins.txt"

    G = load_data.load_graph(edges_path)
    checkins = load_data.load_checkins(checkins_path)

    filtered_graph, filtered_checkins, active_users = filter_active_users(G, checkins)
    stats = compute_basic_graph_stats(filtered_graph)
    logger.info("Graph statistics: %s", stats)

    user_to_locations, location_to_users, user_to_checkins = build_user_location_maps(filtered_checkins)
    logger.info("Lookup maps built:")
    logger.info("users with locations: %d", len(user_to_locations))
    logger.info("locations with users: %d", len(location_to_users))
    logger.info("users with checkins: %d", len(user_to_checkins))

    G_train, train_edges, test_edges = create_train_test_split(filtered_graph)
    logger.info("Train edges: %d", len(train_edges))
    logger.info("Test edges: %d", len(test_edges))
    logger.info("Training graph statistics: %s", compute_basic_graph_stats(G_train))
