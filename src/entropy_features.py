"""Entropy-style features for users/locations based on visit distributions."""
from __future__ import annotations

from typing import Iterable
import math


def shannon_entropy(counts: Iterable[int]) -> float:
    """Compute Shannon entropy given integer counts (not normalized)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent
