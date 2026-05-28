"""Data loading utilities for Gowalla dataset.

Functions:
- load_edges(path) -> pandas.DataFrame
- load_checkins(path) -> pandas.DataFrame
"""
from __future__ import annotations
import pandas as pd
import networkx as nx
import numpy as np


def _downcast_int_series(s: pd.Series) -> pd.Series:
    """Downcast integer-like series to smallest signed integer dtype that fits."""
    s = pd.to_numeric(s, errors="raise", downcast="signed")
    return s


def load_edges(path: str, sep: str = "\t") -> pd.DataFrame:
    """Load an edge list into a DataFrame with columns `u`, `v`.

    This function validates the input and downcasts node ids to integer dtypes
    to reduce memory usage.

    Raises
    - ValueError: if the input file has fewer than 2 columns.
    """
    df = pd.read_csv(path, sep=sep, comment="#", header=None, low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"Edges file '{path}' must contain at least 2 columns (u, v).")

    df = df.rename(columns={0: "u", 1: "v"})

    # Ensure integer dtype for node ids and reduce memory footprint
    df["u"] = _downcast_int_series(df["u"]).astype(np.int32)
    df["v"] = _downcast_int_series(df["v"]).astype(np.int32)

    # Keep only the canonical first two columns
    return df[["u", "v"]]


def load_checkins(path: str, sep: str = "\t") -> pd.DataFrame:
    """Load checkin records into a DataFrame with optimized dtypes.

    The returned DataFrame has columns: `user`, `time`, `lat`, `lon`, `location`.

    Raises
    - ValueError: if the input file has fewer than 5 columns.
    """
    df = pd.read_csv(path, sep=sep, comment="#", header=None, low_memory=False)
    if df.shape[1] < 5:
        raise ValueError(f"Checkins file '{path}' must contain at least 5 columns: user, time, lat, lon, location.")

    df = df.rename(columns={0: "user", 1: "time", 2: "lat", 3: "lon", 4: "location"})

    # Optimize dtypes: users and locations are integers; lat/lon float32
    try:
        df["user"] = pd.to_numeric(df["user"], errors="raise").astype(np.int32)
    except Exception:
        # keep as object if not integer-like
        pass

    try:
        df["location"] = pd.to_numeric(df["location"], errors="raise").astype(np.int32)
    except Exception:
        pass

    # lat/lon to float32 when possible
    try:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").astype(np.float32)
    except Exception:
        pass

    try:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").astype(np.float32)
    except Exception:
        pass

    return df[["user", "time", "lat", "lon", "location"]]


def load_graph(path: str, sep: str = "\t") -> nx.Graph:
    """Load edges from `path` and return an undirected NetworkX graph.

    This calls :func:`load_edges` internally and builds an undirected graph.
    """
    edges = load_edges(path, sep=sep)
    G = nx.Graph()
    # add_edges_from accepts an iterable of (u, v) tuples
    G.add_edges_from(map(tuple, edges[["u", "v"]].to_numpy()))
    return G
