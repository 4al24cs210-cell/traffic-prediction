"""
data_ingestion.py
-----------------
Load, validate, and profile a traffic-flow CSV file.

Public API
----------
    load_data(path)          -> pd.DataFrame
    validate_schema(df)      -> None  (raises ValueError on failure)
    data_quality_report(df)  -> None  (prints summary to stdout)
"""

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Required columns every input CSV must contain.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = ["timestamp", "location_id", "vehicle_count"]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load a traffic CSV and return a validated DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``timestamp`` parsed as *datetime64* and sorted by
        (location_id, timestamp).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Parse timestamp with flexible format recognition (infer_datetime_format removed in pandas 2.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    validate_schema(df)

    # Ensure consistent ordering
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* is missing required columns or has wrong types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # vehicle_count must be numeric
    if not pd.api.types.is_numeric_dtype(df["vehicle_count"]):
        raise ValueError(
            "Column 'vehicle_count' must be numeric. "
            f"Got dtype: {df['vehicle_count'].dtype}"
        )


def data_quality_report(df: pd.DataFrame) -> None:
    """Print a human-readable data-quality summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded (and validated) traffic DataFrame.
    """
    print("=" * 55)
    print("DATA QUALITY REPORT")
    print("=" * 55)
    print(f"  Total rows       : {len(df):,}")
    print(f"  Total columns    : {len(df.columns)}")

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        print(f"  Date range       : {df['timestamp'].min()} -> {df['timestamp'].max()}")

    if "location_id" in df.columns:
        locations = df["location_id"].unique()
        print(f"  Locations ({len(locations):2d})   : {', '.join(sorted(str(l) for l in locations))}")

    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    for col, n in missing.items():
        pct = 100 * n / len(df) if len(df) else 0
        flag = " ⚠" if n > 0 else ""
        print(f"    {col:<25} {n:>6,} ({pct:5.1f}%){flag}")

    dup_rows = df.duplicated().sum()
    print(f"\n  Duplicate rows   : {dup_rows:,}")
    print("=" * 55)
