"""
src/location_data.py
--------------------
Load and query the Karnataka administrative hierarchy from data/locations.csv.

Data source
-----------
Government of India — Local Government Directory (LGD)
https://lgdirectory.gov.in/
Karnataka state code: 29

The bundled data/locations.csv contains:
  - All 31 Karnataka districts (LGD-verified names and codes)
  - All official taluks per district (LGD-verified names)
  - Village column: empty in the seed file — must be populated by
    importing an official LGD village export via scripts/update_locations.py

Public API
----------
    load_locations(path)              -> pd.DataFrame
    get_countries(df)                 -> list[str]
    get_states(df, country)           -> list[str]
    get_districts(df, state)          -> list[str]
    get_sub_districts(df, state, district) -> list[str]
    get_villages(df, state, district, sub_district, query) -> list[str]
    get_location_ids_for_area(mapping_df, **kwargs) -> list[str]
    validate_hierarchy_combination(df, **kwargs) -> tuple[bool, str]
    load_location_id_mapping(path)    -> pd.DataFrame
    resolve_location_id(mapping_df, location_id) -> dict | None
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Default file paths
# ---------------------------------------------------------------------------

LOCATIONS_CSV = os.path.join("data", "locations.csv")
LOCATION_ID_MAPPING_CSV = os.path.join("data", "location_id_mapping.csv")

# Columns expected in locations.csv
REQUIRED_COLS = ["country", "state", "district", "sub_district"]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_locations(path: str = LOCATIONS_CSV) -> pd.DataFrame:
    """Load the location hierarchy CSV.

    Parameters
    ----------
    path : str
        Path to locations.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned, normalised hierarchy DataFrame.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Location data not found at '{path}'. "
            "The seed file should be present at data/locations.csv. "
            "To import official LGD data run: python scripts/update_locations.py"
        )

    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"locations.csv is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    return df


def load_location_id_mapping(path: str = LOCATION_ID_MAPPING_CSV) -> pd.DataFrame:
    """Load the mapping between internal location_id and geographic hierarchy.

    Returns
    -------
    pd.DataFrame  (empty if file not found)
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["location_id", "country", "state", "district",
                                     "sub_district", "village", "notes"])
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()
    return df


# ---------------------------------------------------------------------------
# Hierarchy queries
# ---------------------------------------------------------------------------


def get_countries(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique countries in the dataset."""
    return sorted(df["country"].replace("", pd.NA).dropna().unique().tolist())


def get_states(df: pd.DataFrame, country: str) -> list[str]:
    """Return sorted list of states for *country*."""
    mask = df["country"] == country
    return sorted(df.loc[mask, "state"].replace("", pd.NA).dropna().unique().tolist())


def get_districts(df: pd.DataFrame, state: str) -> list[str]:
    """Return sorted list of districts for *state*."""
    mask = df["state"] == state
    return sorted(df.loc[mask, "district"].replace("", pd.NA).dropna().unique().tolist())


def get_sub_districts(df: pd.DataFrame, state: str, district: str) -> list[str]:
    """Return sorted list of sub-districts / taluks for *district* in *state*."""
    mask = (df["state"] == state) & (df["district"] == district)
    col = "sub_district"
    return sorted(df.loc[mask, col].replace("", pd.NA).dropna().unique().tolist())


def get_villages(
    df: pd.DataFrame,
    state: str,
    district: str,
    sub_district: str,
    query: str = "",
) -> list[str]:
    """Return village names for the given hierarchy, optionally filtered by *query*.

    Parameters
    ----------
    df : pd.DataFrame
    state, district, sub_district : str
    query : str
        Case-insensitive substring filter for village names.

    Returns
    -------
    list[str]
        Sorted village names. Empty list if no village data is loaded.
    """
    mask = (
        (df["state"] == state)
        & (df["district"] == district)
        & (df["sub_district"] == sub_district)
    )
    if "village" not in df.columns:
        return []
    villages = df.loc[mask, "village"].replace("", pd.NA).dropna().unique().tolist()
    if query:
        villages = [v for v in villages if query.lower() in v.lower()]
    return sorted(villages)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_hierarchy_combination(
    df: pd.DataFrame,
    country: str = "",
    state: str = "",
    district: str = "",
    sub_district: str = "",
    village: str = "",
) -> tuple[bool, str]:
    """Check that the given combination exists in the hierarchy.

    Returns
    -------
    (True, "")          — combination is valid
    (False, reason_str) — combination is invalid with explanation
    """
    if not state:
        return False, "State must be selected."
    if not district:
        return False, "District must be selected."

    # Check state belongs to country
    if country:
        valid_states = get_states(df, country)
        if state not in valid_states:
            return False, f"State '{state}' does not belong to country '{country}'."

    # Check district belongs to state
    valid_districts = get_districts(df, state)
    if district not in valid_districts:
        return False, f"District '{district}' does not belong to state '{state}'."

    if sub_district:
        valid_subs = get_sub_districts(df, state, district)
        if sub_district not in valid_subs:
            return False, (
                f"Sub-district/Taluk '{sub_district}' does not belong to "
                f"district '{district}'."
            )

    if village:
        valid_villages = get_villages(df, state, district, sub_district)
        if valid_villages and village not in valid_villages:
            return False, (
                f"Village '{village}' does not belong to "
                f"sub-district '{sub_district}' / district '{district}'."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Location ID resolution
# ---------------------------------------------------------------------------


def resolve_location_id(
    mapping_df: pd.DataFrame,
    location_id: str,
) -> Optional[dict]:
    """Return the geographic record for *location_id*, or None if not mapped.

    Parameters
    ----------
    mapping_df : pd.DataFrame
        Loaded from data/location_id_mapping.csv.
    location_id : str

    Returns
    -------
    dict or None
    """
    rows = mapping_df[mapping_df["location_id"] == location_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def get_location_ids_for_area(
    mapping_df: pd.DataFrame,
    district: str = "",
    sub_district: str = "",
    village: str = "",
) -> list[str]:
    """Return internal location_ids that match the geographic filters.

    Parameters
    ----------
    mapping_df : pd.DataFrame
    district, sub_district, village : str
        If empty, that level is not filtered.

    Returns
    -------
    list[str]
    """
    mask = pd.Series([True] * len(mapping_df), index=mapping_df.index)
    if district:
        mask &= mapping_df["district"].str.strip() == district
    if sub_district:
        mask &= mapping_df["sub_district"].str.strip() == sub_district
    if village:
        mask &= mapping_df.get("village", pd.Series([""] * len(mapping_df))).str.strip() == village
    return mapping_df.loc[mask, "location_id"].tolist()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def location_summary(df: pd.DataFrame) -> dict:
    """Return a dict with counts of loaded hierarchy records."""
    n_districts = df["district"].replace("", pd.NA).dropna().nunique()
    n_sub = df["sub_district"].replace("", pd.NA).dropna().nunique()
    n_vil = df["village"].replace("", pd.NA).dropna().nunique() if "village" in df.columns else 0
    return {
        "districts": n_districts,
        "sub_districts": n_sub,
        "villages": n_vil,
        "total_rows": len(df),
    }
