"""
tests/test_models.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_baseline import LinearRegressionModel, RandomForestModel, build_model
from src.model_interface import TrafficModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_Xy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "vehicle_count_lag_1": rng.uniform(10, 100, n),
            "vehicle_count_lag_2": rng.uniform(10, 100, n),
        }
    )
    y = pd.Series(rng.uniform(10, 100, n), name="vehicle_count")
    return X, y


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

def test_linear_regression_is_traffic_model():
    assert isinstance(LinearRegressionModel(), TrafficModel)


def test_random_forest_is_traffic_model():
    assert isinstance(RandomForestModel(), TrafficModel)


# ---------------------------------------------------------------------------
# LinearRegressionModel
# ---------------------------------------------------------------------------

class TestLinearRegressionModel:
    def test_fit_and_predict_shape(self):
        X, y = _make_Xy()
        model = LinearRegressionModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        model = LinearRegressionModel()
        X, _ = _make_Xy()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predictions_are_numeric(self):
        X, y = _make_Xy()
        model = LinearRegressionModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating)
        assert not np.any(np.isnan(preds))

    def test_save_and_load(self, tmp_path):
        X, y = _make_Xy()
        model = LinearRegressionModel()
        model.fit(X, y)
        path = str(tmp_path / "lr_model.pkl")
        model.save(path)
        assert os.path.exists(path)

        model2 = LinearRegressionModel()
        model2.load(path)
        preds = model2.predict(X)
        assert preds.shape == (len(X),)

    def test_load_missing_raises(self):
        model = LinearRegressionModel()
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/model.pkl")


# ---------------------------------------------------------------------------
# RandomForestModel
# ---------------------------------------------------------------------------

class TestRandomForestModel:
    def test_fit_and_predict_shape(self):
        X, y = _make_Xy()
        model = RandomForestModel(n_estimators=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        model = RandomForestModel()
        X, _ = _make_Xy()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predictions_are_numeric(self):
        X, y = _make_Xy()
        model = RandomForestModel(n_estimators=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_feature_importances_available_after_fit(self):
        X, y = _make_Xy()
        model = RandomForestModel(n_estimators=5)
        model.fit(X, y)
        fi = model.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]

    def test_save_and_load(self, tmp_path):
        X, y = _make_Xy()
        model = RandomForestModel(n_estimators=5)
        model.fit(X, y)
        path = str(tmp_path / "rf_model.pkl")
        model.save(path)

        model2 = RandomForestModel()
        model2.load(path)
        preds = model2.predict(X)
        assert preds.shape == (len(X),)


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------

def test_build_model_linear_regression():
    config = {"model": {"type": "linear_regression"}}
    model = build_model(config)
    assert isinstance(model, LinearRegressionModel)


def test_build_model_random_forest():
    config = {
        "model": {"type": "random_forest"},
        "random_forest": {"n_estimators": 10, "max_depth": 5},
        "training": {"random_seed": 0},
    }
    model = build_model(config)
    assert isinstance(model, RandomForestModel)


def test_build_model_unknown_raises():
    config = {"model": {"type": "xgboost_not_implemented"}}
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(config)
