"""
feature_engineering.py
-----------------------
Create model-ready features from a cleaned traffic-flow DataFrame.

All lag and rolling features are computed **per location_id group** to
prevent cross-location data leakage.

Public API
----------
    add_time_features(df)                       -> pd.DataFrame
    add_lag_features(df, lags, target)          -> pd.DataFrame
    add_rolling_features(df, windows, target)   -> pd.DataFrame
    build_features(df, lags, windows, target)   -> pd.DataFrame
    get_feature_columns(df, target)             -> list[str]
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the ``timestamp`` column.

    Added columns: ``hour``, ``day_of_week``, ``month``, ``is_weekend``,
    ``quarter``, ``hour_sin``, ``hour_cos`` (cyclical encoding of hour).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``timestamp`` column of datetime type.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek        # 0=Mon … 6=Sun
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding so hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    lags: List[int] = None,
    target: str = "vehicle_count",
) -> pd.DataFrame:
    """Add lagged values of *target* as new columns.

    Lags are created per ``location_id`` group to avoid leaking values from
    one sensor into another.

    Parameters
    ----------
    df : pd.DataFrame
    lags : list[int]
        Lag periods (in number of rows / intervals).
        Default: [1, 2, 4, 8].
    target : str
        Column to lag.

    Returns
    -------
    pd.DataFrame
    """
    if lags is None:
        lags = [1, 2, 4, 8]

    df = df.copy()

    if "location_id" in df.columns:
        for lag in lags:
            df[f"{target}_lag_{lag}"] = df.groupby("location_id")[target].shift(lag)
    else:
        for lag in lags:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling / window features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = None,
    target: str = "vehicle_count",
) -> pd.DataFrame:
    """Add rolling mean and rolling std of *target*.

    Windows are applied per ``location_id`` group.

    Parameters
    ----------
    df : pd.DataFrame
    windows : list[int]
        Rolling window sizes (in intervals).
        Default: [4, 12]  (≈ 1 h and 3 h at 15-min intervals).
    target : str

    Returns
    -------
    pd.DataFrame
    """
    if windows is None:
        windows = [4, 12]

    df = df.copy()

    if "location_id" in df.columns:
        for w in windows:
            shifted = df.groupby("location_id")[target].shift(1)
            df[f"{target}_roll_mean_{w}"] = (
                shifted.groupby(df["location_id"]).transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
            )
            df[f"{target}_roll_std_{w}"] = (
                shifted.groupby(df["location_id"]).transform(
                    lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
                )
            )
    else:
        for w in windows:
            df[f"{target}_roll_mean_{w}"] = (
                df[target].shift(1).rolling(window=w, min_periods=1).mean()
            )
            df[f"{target}_roll_std_{w}"] = (
                df[target].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    lags: List[int] = None,
    windows: List[int] = None,
    target: str = "vehicle_count",
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline in sequence.

    Order: time features → lag features → rolling features → drop NaN rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (preprocessed) DataFrame.
    lags : list[int], optional
    windows : list[int], optional
    target : str

    Returns
    -------
    pd.DataFrame
        Feature-complete DataFrame with NaN rows from lag warm-up dropped.
    """
    if lags is None:
        lags = [1, 2, 4, 8]
    if windows is None:
        windows = [4, 12]

    df = add_time_features(df)
    df = add_lag_features(df, lags=lags, target=target)
    df = add_rolling_features(df, windows=windows, target=target)

    # Drop rows with NaN introduced by lag/rolling (warm-up period)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[build_features] Dropped {dropped:,} warm-up rows with NaN features.")

    return df


def get_feature_columns(df: pd.DataFrame, target: str = "vehicle_count") -> List[str]:
    """Return the list of feature columns (exclude metadata and target).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    target : str

    Returns
    -------
    list[str]
        Columns suitable for use as model input ``X``.
    """
    # Exclude metadata, target, and any non-numeric columns (e.g. geo string fields).
    # Geo columns (country, state, district, sub_district) are informational only
    # and must not be fed into the ML model.
    meta_exclude = {"timestamp", "location_id", target,
                    "country", "state", "district", "sub_district", "village", "notes"}
    return [
        c for c in df.columns
        if c not in meta_exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
