"""
tests/test_data_ingestion.py
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

# Adjust path so tests can import src modules when run from project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion import load_data, validate_schema, data_quality_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path, rows=None):
    """Write a minimal valid CSV and return its path."""
    if rows is None:
        rows = [
            {"timestamp": "2023-01-01 00:00:00", "location_id": "LOC_01", "vehicle_count": 10},
            {"timestamp": "2023-01-01 00:15:00", "location_id": "LOC_01", "vehicle_count": 12},
            {"timestamp": "2023-01-01 00:00:00", "location_id": "LOC_02", "vehicle_count": 5},
        ]
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_path, "test_traffic.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_data_returns_dataframe(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_data(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_data_parses_timestamp(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_data(path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_data_sorted_by_location_then_timestamp(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_data(path)
    for loc, grp in df.groupby("location_id"):
        assert grp["timestamp"].is_monotonic_increasing, f"Not sorted for {loc}"


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("/non/existent/path.csv")


def test_validate_schema_raises_on_missing_column(tmp_path):
    df = pd.DataFrame({"timestamp": ["2023-01-01"], "vehicle_count": [5]})  # missing location_id
    with pytest.raises(ValueError, match="location_id"):
        validate_schema(df)


def test_validate_schema_raises_on_non_numeric_count(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01"],
            "location_id": ["LOC_01"],
            "vehicle_count": ["not_a_number"],
        }
    )
    with pytest.raises(ValueError, match="numeric"):
        validate_schema(df)


def test_data_quality_report_runs_without_error(tmp_path, capsys):
    path = _make_csv(str(tmp_path))
    df = load_data(path)
    data_quality_report(df)
    captured = capsys.readouterr()
    assert "Total rows" in captured.out
    assert "Missing values" in captured.out
