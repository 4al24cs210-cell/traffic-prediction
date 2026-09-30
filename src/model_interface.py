"""
model_interface.py
------------------
Abstract base class that all traffic-flow prediction models must implement.

Subclasses
----------
    LinearRegressionModel   (model_baseline.py)
    RandomForestModel       (model_baseline.py)
    LSTMModel               (model_lstm.py — future)
    GRUModel                (model_gru.py  — future)

Any class that inherits TrafficModel and implements all abstract methods
can be plugged into train.py / predict.py by changing config.yaml only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class TrafficModel(ABC):
    """Abstract interface for traffic-flow prediction models."""

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (rows = samples, columns = features).
        y : pd.Series
            Target values (vehicle count at the *next* interval).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run inference on *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            1-D array of predicted vehicle counts, shape (n_samples,).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path*.

        Parameters
        ----------
        path : str
            File path (e.g. ``models/model.pkl``).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved model from *path*.

        Parameters
        ----------
        path : str
            File path to the saved artifact.
        """

    # ------------------------------------------------------------------
    # Optional helpers with default implementations
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name of the model class."""
        return self.__class__.__name__

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name}>"
