"""
scripts/update_locations.py
----------------------------
Import an official LGD (Local Government Directory) export into
data/locations.csv.

OFFICIAL DATA SOURCE
--------------------
Government of India — Local Government Directory (LGD)
URL  : https://lgdirectory.gov.in/
State: Karnataka  (LGD State Code: 29)

Required LGD entity exports (downloadable from the LGD portal):
  1. District Report      — "Download All Districts" for Karnataka
  2. Sub-District Report  — "Download All Sub-Districts" for Karnataka
  3. Village Report       — "Download All Villages" for Karnataka

Expected LGD export column names (LGD uses these headers in XLS/CSV):
  District export  : State Code, State Name, District Code, District Name (In English)
  Sub-dist export  : State Code, State Name, District Code, District Name,
                     Sub-District Code, Sub-District Name (In English)
  Village export   : State Code, State Name, District Code, District Name,
                     Sub-District Code, Sub-District Name,
                     Village Code, Village Name (In English)

Usage
-----
  # Import only districts and sub-districts (no village CSV available):
  python scripts/update_locations.py \\
      --district data/lgd_districts.xlsx \\
      --subdistrict data/lgd_subdistricts.xlsx

  # Import full hierarchy including villages:
  python scripts/update_locations.py \\
      --district data/lgd_districts.xlsx \\
      --subdistrict data/lgd_subdistricts.xlsx \\
      --village data/lgd_villages.xlsx

  # Validate existing data/locations.csv without importing:
  python scripts/update_locations.py --validate-only

Output
------
  data/locations.csv   — updated with official LGD records
  A backup of the previous file is saved to data/locations_backup_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

LOCATIONS_PATH = os.path.join("data", "locations.csv")
DISTRICTS_PATH = os.path.join("data", "karnataka_districts_lgd.csv")

# --------------------------------------------------------------------------
# Column name normalisers — LGD exports use varying header names.
# Map known LGD export headers to our internal schema.
# --------------------------------------------------------------------------

DISTRICT_COL_MAP = {
    # LGD header variants -> internal name
    "State Code": "state_lgd_code",
    "State Name": "state",
    "District Code": "district_lgd_code",
    "District Name (In English)": "district",
    "District Name": "district",
    "DistrictCode": "district_lgd_code",
    "DistrictName": "district",
}

SUBDISTRICT_COL_MAP = {
    "State Code": "state_lgd_code",
    "State Name": "state",
    "District Code": "district_lgd_code",
    "District Name (In English)": "district",
    "District Name": "district",
    "Sub-District Code": "sub_district_lgd_code",
    "Sub-District Name (In English)": "sub_district",
    "Sub-District Name": "sub_district",
    "SubDistrictCode": "sub_district_lgd_code",
    "SubDistrictName": "sub_district",
}

VILLAGE_COL_MAP = {
    "State Code": "state_lgd_code",
    "State Name": "state",
    "District Code": "district_lgd_code",
    "District Name (In English)": "district",
    "District Name": "district",
    "Sub-District Code": "sub_district_lgd_code",
    "Sub-District Name (In English)": "sub_district",
    "Sub-District Name": "sub_district",
    "Village Code": "village_lgd_code",
    "Village Name (In English)": "village",
    "Village Name": "village",
}


def _read_lgd_file(path: str) -> pd.DataFrame:
    """Read an LGD export file (CSV or XLS/XLSX)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)


def _normalise_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename LGD export columns to internal schema names."""
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    df.columns = [c.strip() for c in df.columns]
    return df


def _backup_existing(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.replace(".csv", f"_backup_{ts}.csv")
    shutil.copy2(path, backup)
    print(f"[update_locations] Backup saved -> {backup}")
    return backup


def validate_hierarchy(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages (empty = no errors)."""
    errors: list[str] = []

    # 1. Required columns
    required = ["country", "state", "district", "sub_district"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    # 2. Blank district or sub_district
    blank_district = df["district"].isna() | (df["district"].str.strip() == "")
    if blank_district.any():
        errors.append(f"{blank_district.sum()} rows have blank district name")

    blank_sub = df["sub_district"].isna() | (df["sub_district"].str.strip() == "")
    if blank_sub.any():
        n = blank_sub.sum()
        # Only flag as error if there are no village-less rows (sub_district empty is ok
        # only when village is also empty — means the row is a taluk placeholder)
        village_col = "village" if "village" in df.columns else None
        if village_col:
            real_blanks = blank_sub & df[village_col].notna() & (df[village_col].str.strip() != "")
            if real_blanks.any():
                errors.append(f"{real_blanks.sum()} rows have blank sub_district but non-empty village")
        # else: all blank sub_district rows are top-level taluk stubs — OK

    # 3. Duplicate (district, sub_district, village) combinations
    check_cols = ["district", "sub_district"]
    if "village" in df.columns:
        check_cols.append("village")
    dups = df[df.duplicated(subset=check_cols, keep=False)]
    if not dups.empty:
        n = len(dups) // 2
        errors.append(f"{n} duplicate (district, sub_district[, village]) combinations found")

    # 4. Karnataka state consistency
    non_ka = df[df["state"].str.strip() != "Karnataka"]
    if not non_ka.empty:
        errors.append(
            f"{len(non_ka)} rows have state != 'Karnataka': "
            f"{non_ka['state'].unique().tolist()}"
        )

    return errors


def import_lgd(
    district_path: str | None = None,
    subdistrict_path: str | None = None,
    village_path: str | None = None,
    output_path: str = LOCATIONS_PATH,
    state_filter: str = "Karnataka",
    state_lgd_code: str = "29",
) -> pd.DataFrame:
    """
    Import LGD exports and build a unified locations.csv.

    Parameters
    ----------
    district_path : str, optional
        Path to LGD district export (CSV/XLS).
    subdistrict_path : str, optional
        Path to LGD sub-district export (CSV/XLS).
    village_path : str, optional
        Path to LGD village export (CSV/XLS). May be very large (~30k rows).
    output_path : str
        Destination path for merged locations CSV.
    state_filter : str
        State name to filter (default: Karnataka).
    state_lgd_code : str
        LGD state code to filter (default: '29' for Karnataka).

    Returns
    -------
    pd.DataFrame
        The merged locations DataFrame.
    """
    if district_path is None and subdistrict_path is None and village_path is None:
        print("[update_locations] No input files provided. Nothing to import.")
        return pd.DataFrame()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and filter each level
    # ------------------------------------------------------------------
    dist_df = sub_df = vil_df = None

    if district_path:
        print(f"[update_locations] Reading districts from: {district_path}")
        raw = _read_lgd_file(district_path)
        raw = _normalise_columns(raw, DISTRICT_COL_MAP)
        dist_df = raw[raw.get("state_lgd_code", pd.Series(dtype=str)).astype(str) == state_lgd_code].copy()
        if dist_df.empty:
            dist_df = raw[raw.get("state", pd.Series(dtype=str)).str.strip() == state_filter].copy()
        dist_df["country"] = "India"
        dist_df["state"] = state_filter
        print(f"  -> {len(dist_df)} districts loaded.")

    if subdistrict_path:
        print(f"[update_locations] Reading sub-districts from: {subdistrict_path}")
        raw = _read_lgd_file(subdistrict_path)
        raw = _normalise_columns(raw, SUBDISTRICT_COL_MAP)
        sub_df = raw[raw.get("state_lgd_code", pd.Series(dtype=str)).astype(str) == state_lgd_code].copy()
        if sub_df.empty:
            sub_df = raw[raw.get("state", pd.Series(dtype=str)).str.strip() == state_filter].copy()
        sub_df["country"] = "India"
        sub_df["state"] = state_filter
        print(f"  -> {len(sub_df)} sub-districts loaded.")

    if village_path:
        print(f"[update_locations] Reading villages from: {village_path}")
        raw = _read_lgd_file(village_path)
        raw = _normalise_columns(raw, VILLAGE_COL_MAP)
        vil_df = raw[raw.get("state_lgd_code", pd.Series(dtype=str)).astype(str) == state_lgd_code].copy()
        if vil_df.empty:
            vil_df = raw[raw.get("state", pd.Series(dtype=str)).str.strip() == state_filter].copy()
        vil_df["country"] = "India"
        vil_df["state"] = state_filter
        print(f"  -> {len(vil_df)} villages loaded.")

    # ------------------------------------------------------------------
    # 2. Build unified DataFrame
    # ------------------------------------------------------------------
    SCHEMA_COLS = [
        "country", "state", "district", "sub_district", "village",
        "state_lgd_code", "district_lgd_code", "sub_district_lgd_code",
        "village_lgd_code", "notes",
    ]

    if vil_df is not None and "sub_district" in vil_df.columns:
        # Full hierarchy from village export
        result = vil_df.copy()
        for col in SCHEMA_COLS:
            if col not in result.columns:
                result[col] = ""
        result = result[SCHEMA_COLS]
    elif sub_df is not None:
        # Sub-district level only (no villages)
        result = sub_df.copy()
        result["village"] = ""
        result["notes"] = "LGD taluk — villages require official LGD export"
        for col in SCHEMA_COLS:
            if col not in result.columns:
                result[col] = ""
        result = result[SCHEMA_COLS]
    elif dist_df is not None:
        # District level only
        result = dist_df.copy()
        result["sub_district"] = ""
        result["village"] = ""
        result["sub_district_lgd_code"] = ""
        result["village_lgd_code"] = ""
        result["notes"] = "LGD district — sub-districts require official LGD export"
        for col in SCHEMA_COLS:
            if col not in result.columns:
                result[col] = ""
        result = result[SCHEMA_COLS]
    else:
        print("[update_locations] No usable data found.")
        return pd.DataFrame()

    result = result.fillna("").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------
    errors = validate_hierarchy(result)
    if errors:
        print("[update_locations] VALIDATION WARNINGS:")
        for e in errors:
            print(f"  ! {e}")
    else:
        print("[update_locations] Hierarchy validation: OK")

    # ------------------------------------------------------------------
    # 4. Backup existing file and write new one
    # ------------------------------------------------------------------
    _backup_existing(output_path)
    result.to_csv(output_path, index=False)
    print(f"[update_locations] Saved {len(result):,} records -> {output_path}")

    # Print summary
    n_districts = result["district"].nunique()
    n_subdistricts = result["sub_district"].replace("", pd.NA).dropna().nunique()
    n_villages = result["village"].replace("", pd.NA).dropna().nunique() if "village" in result.columns else 0
    print(f"\n  Summary:")
    print(f"    Districts    : {n_districts}")
    print(f"    Sub-districts: {n_subdistricts}")
    print(f"    Villages     : {n_villages}")

    return result


def validate_only(path: str = LOCATIONS_PATH) -> None:
    """Load and validate existing locations.csv."""
    if not os.path.exists(path):
        print(f"[update_locations] File not found: {path}")
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    errors = validate_hierarchy(df)
    print(f"[update_locations] Validating {path} ({len(df):,} rows)")
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  ! {e}")
    else:
        print("Validation: PASSED")
    n_districts = df["district"].nunique()
    n_sub = df["sub_district"].replace("", pd.NA).dropna().nunique()
    n_vil = df["village"].replace("", pd.NA).dropna().nunique() if "village" in df.columns else 0
    print(f"  Districts    : {n_districts}")
    print(f"  Sub-districts: {n_sub}")
    print(f"  Villages     : {n_vil}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import official LGD data into data/locations.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--district", default=None, help="Path to LGD district export (CSV/XLS)")
    parser.add_argument("--subdistrict", default=None, help="Path to LGD sub-district export (CSV/XLS)")
    parser.add_argument("--village", default=None, help="Path to LGD village export (CSV/XLS)")
    parser.add_argument("--output", default=LOCATIONS_PATH, help=f"Output path (default: {LOCATIONS_PATH})")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing locations.csv")
    args = parser.parse_args()

    if args.validate_only:
        validate_only(args.output)
        return

    import_lgd(
        district_path=args.district,
        subdistrict_path=args.subdistrict,
        village_path=args.village,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
