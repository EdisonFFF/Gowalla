"""Model training and prediction wrappers for experiments."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator


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
