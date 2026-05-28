"""Evaluation metrics for link prediction experiments."""
from __future__ import annotations

from typing import Sequence
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_scores(y_true: Sequence[int], y_score: Sequence[float]) -> dict:
    """Return common binary-class metrics (AUC, AP)."""
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
    }
