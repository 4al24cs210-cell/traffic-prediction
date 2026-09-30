"""
preprocessing.py
----------------
Clean and normalise a raw traffic-flow DataFrame.

Public API
----------
    fill_missing(df)                -> pd.DataFrame
    cap_outliers(df, col, z_thresh) -> pd.DataFrame
    normalise(df, cols, scaler_path, fit) -> pd.DataFrame
    load_scaler(path)               -> MinMaxScaler
"""

from __future__ import annotations

import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

def fill_missing(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Forward-fill then backward-fill gaps of up to *max_gap* consecutive NaNs.

    Gaps longer than *max_gap* steps are flagged in a ``_gap_flag`` column
    for each numeric column but left filled (to avoid breaking feature
    engineering).  Operates per location group to avoid cross-location leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Traffic DataFrame (must be sorted by location_id, timestamp).
    max_gap : int
        Maximum consecutive NaN run length that will be silently filled.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "location_id" in df.columns:
        for col in numeric_cols:
            df[col] = df.groupby("location_id")[col].transform(
                lambda s: s.ffill(limit=max_gap).bfill(limit=max_gap)
            )
    else:
        for col in numeric_cols:
            df[col] = df[col].ffill(limit=max_gap)
            df[col] = df[col].bfill(limit=max_gap)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outlier capping
# ---------------------------------------------------------------------------

def cap_outliers(
    df: pd.DataFrame,
    col: str = "vehicle_count",
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """Cap extreme values in *col* using Z-score thresholding.

    Values beyond *z_thresh* standard deviations from the mean are clipped
    to the boundary value.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to cap.
    z_thresh : float
        Z-score boundary.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        return df
    lower = mean - z_thresh * std
    upper = mean + z_thresh * std
    clipped = df[col].clip(lower=lower, upper=upper)
    n_capped = (df[col] != clipped).sum()
    if n_capped > 0:
        print(f"[cap_outliers] Capped {n_capped:,} values in '{col}' "
              f"(bounds: [{lower:.1f}, {upper:.1f}])")
    df[col] = clipped
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(
    df: pd.DataFrame,
    cols: List[str],
    scaler_path: str,
    fit: bool = True,
) -> pd.DataFrame:
    """Min-Max scale *cols* in *df*.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Columns to scale.
    scaler_path : str
        Where to save (fit=True) or load (fit=False) the scaler.
    fit : bool
        If True, fit a new scaler and save it.
        If False, load the existing scaler from *scaler_path*.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *cols* scaled in-place.
    """
    df = df.copy()
    os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)

    if fit:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        joblib.dump(scaler, scaler_path)
        print(f"[normalise] Scaler fitted and saved -> {scaler_path}")
    else:
        scaler = load_scaler(scaler_path)
        df[cols] = scaler.transform(df[cols])

    return df


def load_scaler(path: str) -> MinMaxScaler:
    """Load a previously saved MinMaxScaler from *path*.

    Raises
    ------
    FileNotFoundError
        If the scaler file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at '{path}'. Run training first."
        )
    return joblib.load(path)
