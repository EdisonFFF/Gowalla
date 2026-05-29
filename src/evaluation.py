"""Evaluation metrics for link prediction experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute ROC AUC."""
    return float(roc_auc_score(y_true, y_score))


def compute_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute precision-recall AUC."""
    return float(average_precision_score(y_true, y_score))


def precision_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
    """Compute precision at rank k for scored candidates."""
    if k <= 0:
        raise ValueError("k must be positive")

    scored = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values(
        "y_score", ascending=False
    )
    top_k = scored.head(k)
    if top_k.empty:
        return 0.0
    return float(top_k["y_true"].mean())


def evaluate_model(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Return the standard link-prediction evaluation summary."""
    results = {
        "AUC": compute_auc(y_true, y_score),
        "PR_AUC": compute_pr_auc(y_true, y_score),
        "P@10": precision_at_k(y_true, y_score, 10),
        "P@50": precision_at_k(y_true, y_score, 50),
        "P@100": precision_at_k(y_true, y_score, 100),
    }
    logger.info("Evaluation results: %s", results)
    return results


def run_ablation_study(results_by_config: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Return a DataFrame of ablation study results for common configurations."""
    configurations = [
        "topology_only",
        "geography_only",
        "entropy_only",
        "topology_geography",
        "all_features",
    ]
    rows = []
    for configuration in configurations:
        metrics = results_by_config.get(configuration, {})
        rows.append(
            {
                "configuration": configuration,
                "AUC": metrics.get("AUC"),
                "PR_AUC": metrics.get("PR_AUC"),
                "P@10": metrics.get("P@10"),
                "P@50": metrics.get("P@50"),
                "P@100": metrics.get("P@100"),
            }
        )

    frame = pd.DataFrame(rows)
    logger.info("Built ablation study table with %d rows", len(frame))
    return frame


def summarize_results(results: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Generate an evaluation table and export it to CSV."""
    output = Path(output_path)
    results.to_csv(output, index=False)
    logger.info("Saved evaluation summary to %s", output)
    return results


def evaluate_scores(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Backward-compatible wrapper for the legacy metrics helper."""
    return {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_pr_auc(y_true, y_score),
    }
