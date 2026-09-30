"""
generate_sample_data.py
-----------------------
Generates a synthetic traffic-flow CSV for development and testing.

Output columns:
    timestamp       - datetime at 15-minute intervals
    location_id     - string identifier for a sensor location (LOC_01 ... LOC_05)
    vehicle_count   - simulated vehicle count (integer >= 0)
    day_of_week     - 0=Monday ... 6=Sunday
    is_holiday      - 1 if a randomly assigned public-holiday, else 0
    temperature     - simulated ambient temperature (degrees C) as an optional feature
    weather_code    - 0=clear, 1=rain, 2=fog  (optional feature)
    country         - always 'India' (geographic mapping, illustrative for demo data)
    state           - always 'Karnataka'
    district        - illustrative district name (see data/location_id_mapping.csv)
    sub_district    - illustrative taluk name
    notes           - reminder that geographic assignments are DEMO only

IMPORTANT: Geographic fields (country/state/district/sub_district) are
DEMO assignments used to link synthetic sensor IDs to the Karnataka
administrative hierarchy.  They are NOT observations from real sensors.
The authoritative location hierarchy is in data/locations.csv.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --output data/raw/custom.csv --days 60
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 96

# Demo geographic mapping for synthetic location IDs.
# Source: data/location_id_mapping.csv
# NOTE: These are illustrative assignments only — not real sensor locations.
LOCATION_GEO = {
    "LOC_01": {"country": "India", "state": "Karnataka", "district": "Bengaluru Urban",  "sub_district": "Bengaluru North"},
    "LOC_02": {"country": "India", "state": "Karnataka", "district": "Mysuru",            "sub_district": "Mysuru"},
    "LOC_03": {"country": "India", "state": "Karnataka", "district": "Dakshina Kannada",  "sub_district": "Mangaluru"},
    "LOC_04": {"country": "India", "state": "Karnataka", "district": "Belagavi",          "sub_district": "Belagavi"},
    "LOC_05": {"country": "India", "state": "Karnataka", "district": "Shivamogga",        "sub_district": "Shivamogga"},
}

LOCATIONS = list(LOCATION_GEO.keys())


# ---------------------------------------------------------------------------
# Traffic profile
# ---------------------------------------------------------------------------
def _daily_profile(rng: np.random.Generator) -> np.ndarray:
    """Return a 96-element array representing intra-day traffic multipliers."""
    profile = np.ones(INTERVALS_PER_DAY) * 0.2  # baseline off-peak
    # Morning rush: ~07:00-09:00  (intervals 28-36)
    profile[28:37] = rng.uniform(0.7, 1.0, 9)
    profile[28] = 0.5
    profile[32] = 1.0   # peak
    profile[36] = 0.75
    # Midday moderate: ~12:00-13:00 (intervals 48-52)
    profile[48:53] = rng.uniform(0.4, 0.6, 5)
    # Evening rush: ~17:00-19:00 (intervals 68-76)
    profile[68:77] = rng.uniform(0.7, 1.0, 9)
    profile[68] = 0.5
    profile[72] = 1.0   # peak
    profile[76] = 0.75
    return profile


def generate(
    output_path: str,
    days: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Build timestamp range
    start = pd.Timestamp("2023-01-01 00:00:00")
    periods = days * INTERVALS_PER_DAY
    timestamps = pd.date_range(start=start, periods=periods, freq=f"{INTERVAL_MINUTES}min")

    # Choose ~10 random holidays
    unique_dates = pd.Series(timestamps.date).unique()
    holiday_dates = set(rng.choice(unique_dates, size=min(10, len(unique_dates)), replace=False))

    records = []
    for loc in LOCATIONS:
        geo = LOCATION_GEO[loc]
        base_volume = rng.integers(80, 200)  # location-level baseline vehicle count at peak
        daily_profile = _daily_profile(rng)

        for ts in timestamps:
            dow = ts.dayofweek          # 0=Mon, 6=Sun
            is_holiday = int(ts.date() in holiday_dates)
            interval_idx = (ts.hour * 60 + ts.minute) // INTERVAL_MINUTES

            # Weekend / holiday suppression
            weekend_factor = 0.5 if dow >= 5 else 1.0
            holiday_factor = 0.3 if is_holiday else 1.0

            # Vehicle count = base x profile x modifiers + noise
            count = (
                base_volume
                * daily_profile[interval_idx]
                * weekend_factor
                * holiday_factor
                + rng.normal(0, base_volume * 0.05)
            )
            count = max(0, int(round(count)))

            # Optional features
            temp = 15 + 10 * np.sin(2 * np.pi * ts.dayofyear / 365) + rng.normal(0, 3)
            weather = int(rng.choice([0, 0, 0, 1, 2], p=[0.7, 0.0, 0.0, 0.2, 0.1]))

            records.append(
                {
                    "timestamp": ts,
                    "location_id": loc,
                    "vehicle_count": count,
                    "day_of_week": dow,
                    "is_holiday": is_holiday,
                    "temperature": round(float(temp), 1),
                    "weather_code": weather,
                    # Geographic fields (illustrative demo mapping only)
                    "country": geo["country"],
                    "state": geo["state"],
                    "district": geo["district"],
                    "sub_district": geo["sub_district"],
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[generate_sample_data] Saved {len(df):,} rows -> {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic traffic data CSV.")
    parser.add_argument(
        "--output",
        default=os.path.join("data", "raw", "sample_traffic.csv"),
        help="Output CSV path (default: data/raw/sample_traffic.csv)",
    )
    parser.add_argument("--days", type=int, default=120, help="Number of days to simulate (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    generate(output_path=args.output, days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
