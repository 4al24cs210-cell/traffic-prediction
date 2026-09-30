"""
tests/test_preprocessing.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import fill_missing, cap_outliers, normalise, load_scaler
from src.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": "LOC_01",
            "vehicle_count": rng.integers(10, 100, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# fill_missing tests
# ---------------------------------------------------------------------------

def test_fill_missing_no_nans_unchanged():
    df = _make_df()
    result = fill_missing(df)
    assert result["vehicle_count"].isna().sum() == 0


def test_fill_missing_fills_short_gap():
    df = _make_df()
    df.loc[5, "vehicle_count"] = np.nan
    result = fill_missing(df)
    assert not result["vehicle_count"].isna().any()


def test_fill_missing_single_row():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01")],
            "location_id": ["LOC_01"],
            "vehicle_count": [np.nan],
        }
    )
    # Single NaN row — bfill/ffill cannot fill it; should not raise
    result = fill_missing(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# cap_outliers tests
# ---------------------------------------------------------------------------

def test_cap_outliers_reduces_extremes():
    df = _make_df()
    df.loc[0, "vehicle_count"] = 999_999  # extreme outlier
    result = cap_outliers(df, col="vehicle_count", z_thresh=3.0)
    assert result["vehicle_count"].max() < 999_999


def test_cap_outliers_zero_std_safe():
    df = _make_df()
    df["vehicle_count"] = 50.0  # all same → std == 0
    result = cap_outliers(df, col="vehicle_count")
    assert (result["vehicle_count"] == 50.0).all()


# ---------------------------------------------------------------------------
# normalise tests
# ---------------------------------------------------------------------------

def test_normalise_fit_saves_scaler(tmp_path):
    df = _make_df()
    scaler_path = str(tmp_path / "scaler.pkl")
    result = normalise(df, cols=["vehicle_count"], scaler_path=scaler_path, fit=True)
    assert os.path.exists(scaler_path)
    assert result["vehicle_count"].min() >= 0.0
    assert result["vehicle_count"].max() <= 1.0


def test_normalise_load_and_transform(tmp_path):
    df = _make_df()
    scaler_path = str(tmp_path / "scaler.pkl")
    normalise(df, cols=["vehicle_count"], scaler_path=scaler_path, fit=True)
    df2 = _make_df()
    result = normalise(df2, cols=["vehicle_count"], scaler_path=scaler_path, fit=False)
    assert result["vehicle_count"].dtype == float


def test_load_scaler_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_scaler("/nonexistent/scaler.pkl")


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

def test_add_time_features_columns():
    df = _make_df()
    result = add_time_features(df)
    for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
        assert col in result.columns


def test_add_lag_features_correct_count():
    df = _make_df()
    lags = [1, 2, 4]
    result = add_lag_features(df, lags=lags)
    for lag in lags:
        assert f"vehicle_count_lag_{lag}" in result.columns


def test_lag_features_per_location_no_leakage():
    """Lag values must not bleed from one location into another."""
    df1 = _make_df(n=10, seed=1)
    df1["location_id"] = "LOC_A"
    df2 = _make_df(n=10, seed=2)
    df2["location_id"] = "LOC_B"
    df = pd.concat([df1, df2], ignore_index=True).sort_values(["location_id", "timestamp"])
    result = add_lag_features(df, lags=[1])
    # First row of each location should be NaN for lag_1
    for loc in ["LOC_A", "LOC_B"]:
        first_idx = result[result["location_id"] == loc].index[0]
        assert pd.isna(result.loc[first_idx, "vehicle_count_lag_1"])


def test_build_features_drops_nan_rows():
    df = _make_df(n=100)
    result = build_features(df)
    assert result.isna().sum().sum() == 0
    assert len(result) < 100  # warm-up rows dropped


def test_get_feature_columns_excludes_metadata():
    df = _make_df(n=50)
    df = build_features(df)
    feat_cols = get_feature_columns(df)
    assert "timestamp" not in feat_cols
    assert "location_id" not in feat_cols
    assert "vehicle_count" not in feat_cols
    assert len(feat_cols) > 0
