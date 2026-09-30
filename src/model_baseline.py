"""
model_baseline.py
-----------------
Baseline traffic-flow prediction models.

Classes
-------
    LinearRegressionModel   — wraps sklearn LinearRegression
    RandomForestModel       — wraps sklearn RandomForestRegressor

Both implement the TrafficModel interface and can be saved/loaded with joblib.

Config keys consumed (via config.yaml → random_forest section):
    random_forest.n_estimators  (default 100)
    random_forest.max_depth     (default 10)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.model_interface import TrafficModel


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

class LinearRegressionModel(TrafficModel):
    """Ordinary least-squares linear regression baseline."""

    def __init__(self) -> None:
        self._model: Optional[LinearRegression] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = LinearRegression()
        self._model.fit(X, y)
        print(f"[{self.name}] Training complete. Coefficients shape: {self._model.coef_.shape}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self._model, path)
        print(f"[{self.name}] Model saved -> {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = joblib.load(path)
        print(f"[{self.name}] Model loaded <- {path}")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(TrafficModel):
    """Random Forest regressor — stronger baseline with non-linear capacity."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
    ) -> None:
        self._model: Optional[RandomForestRegressor] = None
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        print(
            f"[{self.name}] Training complete. "
            f"Trees: {self._n_estimators}, max_depth: {self._max_depth}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self._model, path)
        print(f"[{self.name}] Model saved -> {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = joblib.load(path)
        print(f"[{self.name}] Model loaded <- {path}")

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Expose feature importances if model is fitted."""
        if self._model is not None:
            return self._model.feature_importances_
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
}


def build_model(config: Dict[str, Any]) -> TrafficModel:
    """Instantiate the model specified in *config*.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml`` dict.

    Returns
    -------
    TrafficModel

    Raises
    ------
    ValueError
        If ``model.type`` is not a known model key.
    """
    model_type = config.get("model", {}).get("type", "random_forest")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    if model_type == "random_forest":
        rf_cfg = config.get("random_forest", {})
        return RandomForestModel(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 10),
            random_state=config.get("training", {}).get("random_seed", 42),
        )

    return MODEL_REGISTRY[model_type]()
