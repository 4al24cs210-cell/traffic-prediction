"""
tests/test_locations.py
-----------------------
Tests for the Karnataka LGD administrative hierarchy data and location utilities.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.location_data import (
    load_locations,
    load_location_id_mapping,
    get_countries,
    get_states,
    get_districts,
    get_sub_districts,
    get_villages,
    validate_hierarchy_combination,
    get_location_ids_for_area,
    location_summary,
    LOCATIONS_CSV,
    LOCATION_ID_MAPPING_CSV,
)
from scripts.update_locations import validate_hierarchy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loc_df():
    """Load the actual bundled locations.csv."""
    return load_locations(LOCATIONS_CSV)


@pytest.fixture(scope="module")
def mapping_df():
    """Load the actual bundled location_id_mapping.csv."""
    return load_location_id_mapping(LOCATION_ID_MAPPING_CSV)


@pytest.fixture
def minimal_loc_df():
    """A minimal test DataFrame with known hierarchy."""
    return pd.DataFrame([
        {"country": "India", "state": "Karnataka", "district": "Mysuru",
         "sub_district": "Mysuru", "village": "Ashokapuram",
         "state_lgd_code": "29", "district_lgd_code": "594",
         "sub_district_lgd_code": "1143", "village_lgd_code": "9001", "notes": ""},
        {"country": "India", "state": "Karnataka", "district": "Mysuru",
         "sub_district": "Nanjanagudu", "village": "Doddahundi",
         "state_lgd_code": "29", "district_lgd_code": "594",
         "sub_district_lgd_code": "1144", "village_lgd_code": "9002", "notes": ""},
        {"country": "India", "state": "Karnataka", "district": "Udupi",
         "sub_district": "Udupi", "village": "",
         "state_lgd_code": "29", "district_lgd_code": "599",
         "sub_district_lgd_code": "1175", "village_lgd_code": "", "notes": "test"},
    ])


# ---------------------------------------------------------------------------
# Section 1: Data file existence and schema
# ---------------------------------------------------------------------------

class TestLocationFileSchema:
    def test_locations_csv_exists(self):
        assert os.path.exists(LOCATIONS_CSV), (
            f"data/locations.csv not found. Ensure the seed file is committed."
        )

    def test_location_mapping_csv_exists(self):
        assert os.path.exists(LOCATION_ID_MAPPING_CSV), (
            f"data/location_id_mapping.csv not found."
        )

    def test_required_columns_present(self, loc_df):
        for col in ["country", "state", "district", "sub_district"]:
            assert col in loc_df.columns, f"Column '{col}' missing from locations.csv"

    def test_lgd_code_columns_present(self, loc_df):
        for col in ["state_lgd_code", "district_lgd_code", "sub_district_lgd_code"]:
            assert col in loc_df.columns, f"Column '{col}' missing from locations.csv"

    def test_no_completely_empty_district(self, loc_df):
        blank = loc_df["district"].replace("", pd.NA).isna()
        assert not blank.any(), f"{blank.sum()} rows have empty district name"

    def test_no_completely_empty_sub_district(self, loc_df):
        # In the seed file sub_district should always be filled
        blank = loc_df["sub_district"].replace("", pd.NA).isna()
        assert not blank.any(), f"{blank.sum()} rows have empty sub_district name"


# ---------------------------------------------------------------------------
# Section 2: Country and State filtering
# ---------------------------------------------------------------------------

class TestCountryStateFilter:
    def test_india_is_the_only_country(self, loc_df):
        countries = get_countries(loc_df)
        assert countries == ["India"], f"Expected only India, got: {countries}"

    def test_karnataka_is_present(self, loc_df):
        states = get_states(loc_df, "India")
        assert "Karnataka" in states

    def test_get_states_unknown_country_returns_empty(self, loc_df):
        states = get_states(loc_df, "Narnia")
        assert states == []

    def test_state_filter_returns_only_karnataka(self, loc_df):
        states = get_states(loc_df, "India")
        assert all(s == "Karnataka" for s in states), (
            f"Non-Karnataka state found: {[s for s in states if s != 'Karnataka']}"
        )


# ---------------------------------------------------------------------------
# Section 3: District filtering
# ---------------------------------------------------------------------------

class TestDistrictFilter:
    def test_karnataka_has_31_districts(self, loc_df):
        districts = get_districts(loc_df, "Karnataka")
        assert len(districts) == 31, (
            f"Expected 31 Karnataka districts, found {len(districts)}: {districts}"
        )

    def test_all_known_districts_present(self, loc_df):
        districts = get_districts(loc_df, "Karnataka")
        # A sample of known LGD-verified districts
        known = [
            "Bengaluru Urban", "Mysuru", "Dakshina Kannada",
            "Belagavi", "Shivamogga", "Udupi", "Kalaburagi",
            "Bagalkote", "Yadgir", "Vijayanagara",
        ]
        for d in known:
            assert d in districts, f"Known district '{d}' not found in locations data"

    def test_get_districts_wrong_state_returns_empty(self, loc_df):
        districts = get_districts(loc_df, "Maharashtra")
        assert districts == []

    def test_district_belongs_to_state(self, minimal_loc_df):
        districts = get_districts(minimal_loc_df, "Karnataka")
        assert "Mysuru" in districts
        assert "Udupi" in districts

    def test_no_district_cross_contamination(self, minimal_loc_df):
        """Districts from one state must not appear under another state."""
        districts = get_districts(minimal_loc_df, "Telangana")
        assert "Mysuru" not in districts


# ---------------------------------------------------------------------------
# Section 4: Taluk / Sub-district filtering
# ---------------------------------------------------------------------------

class TestSubDistrictFilter:
    def test_mysuru_has_taluks(self, loc_df):
        taluks = get_sub_districts(loc_df, "Karnataka", "Mysuru")
        assert len(taluks) >= 5, f"Expected >= 5 taluks for Mysuru, got {len(taluks)}"

    def test_taluk_belongs_to_correct_district(self, minimal_loc_df):
        taluks = get_sub_districts(minimal_loc_df, "Karnataka", "Mysuru")
        assert "Mysuru" in taluks
        assert "Nanjanagudu" in taluks

    def test_taluk_does_not_leak_across_districts(self, minimal_loc_df):
        taluks_mysuru = get_sub_districts(minimal_loc_df, "Karnataka", "Mysuru")
        taluks_udupi  = get_sub_districts(minimal_loc_df, "Karnataka", "Udupi")
        assert "Mysuru" not in taluks_udupi
        assert "Udupi" not in taluks_mysuru

    def test_unknown_district_returns_empty_taluks(self, loc_df):
        taluks = get_sub_districts(loc_df, "Karnataka", "NonexistentDistrict")
        assert taluks == []

    def test_all_districts_have_at_least_one_taluk(self, loc_df):
        for district in get_districts(loc_df, "Karnataka"):
            taluks = get_sub_districts(loc_df, "Karnataka", district)
            assert len(taluks) >= 1, f"District '{district}' has no taluks"


# ---------------------------------------------------------------------------
# Section 5: Village filtering
# ---------------------------------------------------------------------------

class TestVillageFilter:
    def test_village_belongs_to_correct_taluk(self, minimal_loc_df):
        villages = get_villages(minimal_loc_df, "Karnataka", "Mysuru", "Mysuru")
        assert "Ashokapuram" in villages

    def test_village_does_not_leak_to_other_taluk(self, minimal_loc_df):
        villages = get_villages(minimal_loc_df, "Karnataka", "Mysuru", "Nanjanagudu")
        assert "Ashokapuram" not in villages

    def test_village_search_filters_correctly(self, minimal_loc_df):
        results = get_villages(minimal_loc_df, "Karnataka", "Mysuru", "Mysuru", query="ash")
        assert "Ashokapuram" in results
        results_none = get_villages(minimal_loc_df, "Karnataka", "Mysuru", "Mysuru", query="xyz_not_exist")
        assert results_none == []

    def test_no_village_data_returns_empty(self, loc_df):
        # Seed file has no village records — all village cells are empty
        villages = get_villages(loc_df, "Karnataka", "Mysuru", "Mysuru")
        assert isinstance(villages, list)
        # May be empty (no villages loaded yet) — should NOT raise


# ---------------------------------------------------------------------------
# Section 6: Hierarchy validation
# ---------------------------------------------------------------------------

class TestHierarchyValidation:
    def test_valid_combination_passes(self, loc_df):
        # Use a known district + taluk from the seed file
        dist = get_districts(loc_df, "Karnataka")[0]
        sub  = get_sub_districts(loc_df, "Karnataka", dist)[0]
        valid, reason = validate_hierarchy_combination(
            loc_df, country="India", state="Karnataka", district=dist, sub_district=sub
        )
        assert valid, f"Expected valid, got: {reason}"

    def test_missing_state_fails(self, loc_df):
        valid, reason = validate_hierarchy_combination(loc_df, country="India", state="")
        assert not valid

    def test_wrong_district_for_state_fails(self, loc_df):
        valid, reason = validate_hierarchy_combination(
            loc_df, country="India", state="Karnataka", district="Mumbai Suburban"
        )
        assert not valid
        assert "Mumbai Suburban" in reason

    def test_wrong_taluk_for_district_fails(self, minimal_loc_df):
        valid, reason = validate_hierarchy_combination(
            minimal_loc_df, state="Karnataka", district="Mysuru", sub_district="Puttur"
        )
        assert not valid
        assert "Puttur" in reason

    def test_wrong_village_for_taluk_fails(self, minimal_loc_df):
        valid, reason = validate_hierarchy_combination(
            minimal_loc_df, state="Karnataka", district="Mysuru",
            sub_district="Mysuru", village="DoesNotExist"
        )
        assert not valid

    def test_missing_district_fails(self, loc_df):
        valid, reason = validate_hierarchy_combination(
            loc_df, state="Karnataka", district=""
        )
        assert not valid


# ---------------------------------------------------------------------------
# Section 7: Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_no_duplicates_in_seed_file(self, loc_df):
        dups = loc_df.duplicated(subset=["district", "sub_district"], keep=False)
        assert not dups.any(), (
            f"{dups.sum()} duplicate (district, sub_district) rows found in locations.csv"
        )

    def test_validate_hierarchy_catches_duplicates(self):
        df_with_dups = pd.DataFrame([
            {"country": "India", "state": "Karnataka", "district": "Mysuru",
             "sub_district": "Mysuru", "village": ""},
            {"country": "India", "state": "Karnataka", "district": "Mysuru",
             "sub_district": "Mysuru", "village": ""},
        ])
        errors = validate_hierarchy(df_with_dups)
        dup_errors = [e for e in errors if "duplicate" in e.lower()]
        assert len(dup_errors) > 0


# ---------------------------------------------------------------------------
# Section 8: Location code consistency
# ---------------------------------------------------------------------------

class TestLocationCodeConsistency:
    def test_state_lgd_code_is_29_for_all_rows(self, loc_df):
        if "state_lgd_code" in loc_df.columns:
            non_29 = loc_df[loc_df["state_lgd_code"].replace("", pd.NA).dropna() != "29"]
            assert non_29.empty, f"{len(non_29)} rows have state_lgd_code != 29"

    def test_district_lgd_codes_are_unique_per_district(self, loc_df):
        if "district_lgd_code" not in loc_df.columns:
            return
        code_map = (
            loc_df[loc_df["district_lgd_code"].replace("", pd.NA).notna()]
            .groupby("district")["district_lgd_code"]
            .nunique()
        )
        multi_coded = code_map[code_map > 1]
        assert multi_coded.empty, (
            f"These districts have multiple LGD codes: {multi_coded.to_dict()}"
        )


# ---------------------------------------------------------------------------
# Section 9: Location ID mapping
# ---------------------------------------------------------------------------

class TestLocationIDMapping:
    def test_mapping_file_loads(self, mapping_df):
        assert isinstance(mapping_df, pd.DataFrame)

    def test_all_demo_location_ids_present(self, mapping_df):
        for loc_id in ["LOC_01", "LOC_02", "LOC_03", "LOC_04", "LOC_05"]:
            assert loc_id in mapping_df["location_id"].values, (
                f"Expected demo location ID '{loc_id}' in mapping file"
            )

    def test_mapped_districts_exist_in_locations(self, loc_df, mapping_df):
        """Every district in the mapping must also exist in locations.csv."""
        mapped_districts = mapping_df["district"].replace("", pd.NA).dropna().unique()
        loc_districts = set(get_districts(loc_df, "Karnataka"))
        for d in mapped_districts:
            assert d in loc_districts, (
                f"Mapped district '{d}' not found in locations.csv"
            )

    def test_get_location_ids_for_area_filters_correctly(self, mapping_df):
        ids = get_location_ids_for_area(mapping_df, district="Mysuru")
        assert "LOC_02" in ids

    def test_get_location_ids_unknown_district_returns_empty(self, mapping_df):
        ids = get_location_ids_for_area(mapping_df, district="UnknownXYZ")
        assert ids == []


# ---------------------------------------------------------------------------
# Section 10: Location summary
# ---------------------------------------------------------------------------

class TestLocationSummary:
    def test_summary_keys_present(self, loc_df):
        s = location_summary(loc_df)
        assert "districts" in s
        assert "sub_districts" in s
        assert "villages" in s
        assert "total_rows" in s

    def test_summary_district_count(self, loc_df):
        s = location_summary(loc_df)
        assert s["districts"] == 31, (
            f"Expected 31 districts in summary, got {s['districts']}"
        )

    def test_summary_subdistricts_nonzero(self, loc_df):
        s = location_summary(loc_df)
        assert s["sub_districts"] > 0
