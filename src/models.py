"""Model training and prediction wrappers for experiments."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Simple wrapper around scikit-learn estimators to standardize API."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> Any:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X)
        return self.estimator.predict(X)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train a reproducible logistic regression model."""
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver="liblinear",
    )
    model.fit(X_train, y_train)
    logger.info("Trained LogisticRegression on %d samples", len(X_train))
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a reproducible random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Trained RandomForestClassifier on %d samples", len(X_train))
    return model


def predict_scores(model: BaseEstimator, X_test: np.ndarray) -> np.ndarray:
    """Return continuous scores suitable for ranking and evaluation."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
        return scores[:, 1] if scores.ndim == 2 and scores.shape[1] > 1 else scores.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test))
    return np.asarray(model.predict(X_test))


def feature_importance(model: BaseEstimator, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return feature importances or coefficients as name-score pairs."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coefficients[0] if coefficients.ndim > 1 else coefficients)
    else:
        raise ValueError("model does not expose feature importances or coefficients")

    if len(feature_names) != len(importances):
        raise ValueError("feature_names length does not match model output")

    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    logger.info("Computed feature importance for %d features", len(ranked))
    return ranked
