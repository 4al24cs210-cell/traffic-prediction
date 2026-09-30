"""
dashboard.py
------------
Streamlit dashboard for the Traffic Flow Prediction Model.
Karnataka geographic hierarchy powered by LGD-verified location data.

Tabs
----
1. Data Explorer        - raw traffic data with location-aware charts
2. Predict Traffic      - hierarchical location selection + prediction form
3. Model Performance    - evaluation metrics and plots

Run
---
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import load_data
from src.predict import _load_config, run_inference, load_model
from src.preprocessing import load_scaler
from src.feature_engineering import build_features, get_feature_columns
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
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Traffic Flow Prediction — Karnataka",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = "config.yaml"
DEFAULT_DATA_PATH = os.path.join("data", "raw", "sample_traffic.csv")
LOCATIONS_CSV = os.path.join("data", "locations.csv")
MAPPING_CSV = os.path.join("data", "location_id_mapping.csv")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_LABELS = {0: "Clear", 1: "Rain", 2: "Fog"}

# Traffic level thresholds (on normalised 0-1 scale).
# These are indicative; document them clearly in the UI.
TRAFFIC_THRESHOLDS = [
    (0.75, "Very High", "#DC2626"),
    (0.50, "High",      "#F59E0B"),
    (0.25, "Moderate",  "#3B82F6"),
    (0.0,  "Low",       "#10B981"),
]


def _classify_traffic(normalised_value: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a normalised traffic value."""
    for threshold, label, colour in TRAFFIC_THRESHOLDS:
        if normalised_value >= threshold:
            return label, colour
    return "Low", "#10B981"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_traffic_csv(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_data(show_spinner=False)
def _load_loc_df() -> pd.DataFrame:
    return load_locations(LOCATIONS_CSV)


@st.cache_data(show_spinner=False)
def _load_mapping_df() -> pd.DataFrame:
    return load_location_id_mapping(MAPPING_CSV)


@st.cache_data(show_spinner=False)
def _load_cfg() -> dict:
    if os.path.exists(CONFIG_PATH):
        return _load_config(CONFIG_PATH)
    return {}


def _model_is_trained(cfg: dict) -> bool:
    return (
        os.path.exists(cfg.get("model", {}).get("save_path", ""))
        and bool(cfg.get("features", {}).get("columns"))
    )


def _load_metrics(cfg: dict) -> dict | None:
    path = cfg.get("reports", {}).get("metrics_path", "reports/metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Sidebar — Geographic Hierarchy + Data Source
# ---------------------------------------------------------------------------

def _sidebar() -> dict:
    """Render sidebar and return a context dict with all selections."""
    st.sidebar.title("🚦 Traffic Flow")
    st.sidebar.markdown("---")

    ctx: dict = {}

    # ---- Location hierarchy ------------------------------------------------
    st.sidebar.subheader("Location Filter")

    try:
        loc_df = _load_loc_df()
        summary = location_summary(loc_df)
        st.sidebar.caption(
            f"LGD data: {summary['districts']} districts, "
            f"{summary['sub_districts']} taluks, "
            f"{summary['villages']} villages loaded"
        )
    except FileNotFoundError as exc:
        st.sidebar.error(str(exc))
        return ctx

    countries = ["All"] + get_countries(loc_df)
    country = st.sidebar.selectbox("Country", countries, index=1 if "India" in countries else 0)

    if country == "All":
        states = ["All"] + sorted(loc_df["state"].replace("", pd.NA).dropna().unique().tolist())
    else:
        states = ["All"] + get_states(loc_df, country)
    state = st.sidebar.selectbox("State", states, index=1 if "Karnataka" in states else 0)

    if state == "All":
        districts = ["All"]
    else:
        districts = ["All"] + get_districts(loc_df, state)
    district = st.sidebar.selectbox("District", districts)

    if district == "All" or state == "All":
        sub_districts = ["All"]
    else:
        sub_districts = ["All"] + get_sub_districts(loc_df, state, district)
    sub_district = st.sidebar.selectbox("Taluk / Sub-District", sub_districts)

    # Village search (only shown when sub_district is selected)
    village = ""
    if sub_district != "All" and district != "All" and state != "All":
        village_query = st.sidebar.text_input("Village search (optional)", placeholder="Type part of village name...")
        village_list = get_villages(loc_df, state, district, sub_district, query=village_query)
        if village_list:
            village = st.sidebar.selectbox("Village / Area", ["(All villages in taluk)"] + village_list)
            if village == "(All villages in taluk)":
                village = ""
        else:
            st.sidebar.caption(
                "No village records loaded yet. "
                "Import official LGD village data:\n"
                "`python scripts/update_locations.py --village <lgd_villages.xlsx>`"
            )

    ctx["country"] = country
    ctx["state"] = state
    ctx["district"] = district
    ctx["sub_district"] = sub_district
    ctx["village"] = village

    # ---- Data source -------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Traffic Data Source")
    use_default = st.sidebar.checkbox("Use sample dataset", value=True)
    data_path = DEFAULT_DATA_PATH

    if not use_default:
        uploaded = st.sidebar.file_uploader("Upload traffic CSV", type=["csv"])
        if uploaded is not None:
            tmp_path = os.path.join("data", "raw", "_uploaded.csv")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as fh:
                fh.write(uploaded.read())
            data_path = tmp_path

    if not os.path.exists(data_path):
        st.sidebar.warning(
            "Sample data not found.\n\n"
            "`python scripts/generate_sample_data.py --days 30`"
        )
        ctx["df"] = None
        ctx["data_path"] = None
    else:
        df_full = _load_traffic_csv(data_path)

        # Apply geographic filter using location_id mapping
        mapping_df = _load_mapping_df()
        if district != "All":
            matched_ids = get_location_ids_for_area(
                mapping_df,
                district=district if district != "All" else "",
                sub_district=sub_district if sub_district != "All" else "",
                village=village,
            )
            if matched_ids:
                df_filtered = df_full[df_full["location_id"].isin(matched_ids)].copy()
            else:
                # Fall back to geographic columns on the traffic data itself
                mask = pd.Series([True] * len(df_full), index=df_full.index)
                if district != "All" and "district" in df_full.columns:
                    mask &= df_full["district"] == district
                if sub_district != "All" and "sub_district" in df_full.columns:
                    mask &= df_full["sub_district"] == sub_district
                df_filtered = df_full[mask].copy()
        else:
            df_filtered = df_full.copy()

        ctx["df"] = df_filtered
        ctx["df_full"] = df_full
        ctx["data_path"] = data_path

    # Date range filter
    if ctx.get("df") is not None and not ctx["df"].empty:
        df = ctx["df"]
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Date range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_date, max_date
        mask = (df["timestamp"].dt.date >= start_d) & (df["timestamp"].dt.date <= end_d)
        ctx["df"] = df[mask].copy()
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Rows shown: {len(ctx['df']):,}")

    return ctx


# ---------------------------------------------------------------------------
# Tab 1 — Data Explorer
# ---------------------------------------------------------------------------

def _tab_data_explorer(ctx: dict):
    st.header("Data Explorer")

    df = ctx.get("df")
    if df is None or (hasattr(df, "empty") and df.empty):
        st.warning("No data to display. Check your data source and location filters.")
        return

    # Location label
    loc_parts = [
        p for p in [ctx.get("country"), ctx.get("state"), ctx.get("district"),
                    ctx.get("sub_district"), ctx.get("village")]
        if p and p not in ("All", "")
    ]
    loc_label = " > ".join(loc_parts) if loc_parts else "All Locations"

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Sensors", df["location_id"].nunique())
    c3.metric("From", str(df["timestamp"].min().date()))
    c4.metric("To", str(df["timestamp"].max().date()))

    st.caption(f"Location filter: **{loc_label}**")
    st.divider()

    # --- Chart 1: Traffic Flow Over Time ---
    st.subheader("Traffic Flow Over Time")
    plot_df = df.groupby("timestamp")["vehicle_count"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(plot_df["timestamp"], plot_df["vehicle_count"], linewidth=0.9, color="#2563EB")
    ax.set_title(f"Traffic Flow Over Time — {loc_label}", fontsize=12)
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Vehicle Count (avg across sensors)")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_a, col_b = st.columns(2)

    # --- Chart 2: Traffic by Hour ---
    with col_a:
        st.subheader("Avg Traffic by Hour of Day")
        hourly = df.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly_agg = hourly.groupby("hour")["vehicle_count"].mean().reindex(range(24), fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.bar(hourly_agg.index, hourly_agg.values, color="#3B82F6", edgecolor="white")
        ax2.set_title("Avg Traffic by Hour (0-23)", fontsize=11)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Avg Vehicle Count")
        ax2.set_xticks(range(24))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Chart 3: Traffic by Day of Week ---
    with col_b:
        st.subheader("Avg Traffic by Day of Week")
        daily = df.copy()
        daily["dow"] = daily["timestamp"].dt.dayofweek
        dow_agg = daily.groupby("dow")["vehicle_count"].mean().reindex(range(7), fill_value=0)
        colours = ["#3B82F6"] * 5 + ["#F59E0B"] * 2  # weekdays blue, weekend amber
        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
        ax3.bar(range(7), dow_agg.values, color=colours, edgecolor="white")
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(DAY_NAMES, rotation=30, ha="right", fontsize=9)
        ax3.set_title("Avg Traffic by Day of Week", fontsize=11)
        ax3.set_ylabel("Avg Vehicle Count")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # --- Location-based traffic statistics ---
    st.subheader("Location-Based Traffic Statistics")
    if "location_id" in df.columns:
        loc_stats = (
            df.groupby("location_id")["vehicle_count"]
            .agg(["mean", "max", "min", "std"])
            .rename(columns={"mean": "Avg", "max": "Peak", "min": "Min", "std": "Std Dev"})
            .reset_index()
        )
        # Enrich with geo info from mapping
        mapping_df = _load_mapping_df()
        if not mapping_df.empty:
            loc_stats = loc_stats.merge(
                mapping_df[["location_id", "district", "sub_district"]],
                on="location_id", how="left"
            )
        st.dataframe(loc_stats, use_container_width=True)

    with st.expander("Show raw data table (first 200 rows)"):
        st.dataframe(df.head(200), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Predict Traffic
# ---------------------------------------------------------------------------

def _tab_predict(ctx: dict):
    st.header("Predict Traffic Flow")

    cfg = _load_cfg()

    # ---- Location selector (hierarchical within the tab) -------------------
    st.subheader("Select Location")

    try:
        loc_df = _load_loc_df()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    col1, col2 = st.columns(2)
    with col1:
        pred_country = st.selectbox("Country", ["India"], key="pred_country")
        pred_state   = st.selectbox("State", ["Karnataka"], key="pred_state")
    with col2:
        districts = get_districts(loc_df, pred_state)
        pred_district = st.selectbox("District", ["— Select —"] + districts, key="pred_district")
        if pred_district == "— Select —":
            st.info("Select a district to continue.")
            return

        sub_districts = get_sub_districts(loc_df, pred_state, pred_district)
        pred_sub = st.selectbox("Taluk / Sub-District", ["— Select —"] + sub_districts, key="pred_sub")
        if pred_sub == "— Select —":
            st.info("Select a taluk to continue.")
            return

    # Village search
    vil_query = st.text_input("Village search (optional)", placeholder="Type part of village name...", key="pred_vil_q")
    villages = get_villages(loc_df, pred_state, pred_district, pred_sub, query=vil_query)
    if villages:
        pred_village = st.selectbox("Village / Area", ["(All villages in taluk)"] + villages, key="pred_village")
        if pred_village == "(All villages in taluk)":
            pred_village = ""
    else:
        pred_village = ""
        st.caption(
            "Village records not yet loaded. "
            "Import official LGD village CSV: `python scripts/update_locations.py --village <file>`"
        )

    # Validate hierarchy
    valid, reason = validate_hierarchy_combination(
        loc_df, country=pred_country, state=pred_state,
        district=pred_district, sub_district=pred_sub, village=pred_village,
    )
    if not valid:
        st.error(f"Invalid location: {reason}")
        return

    location_display = " > ".join(filter(None, [
        pred_country, pred_state, pred_district, pred_sub, pred_village
    ]))

    st.success(f"Location: **{location_display}**")
    st.divider()

    # ---- Check model is trained -------------------------------------------
    if not _model_is_trained(cfg):
        st.warning(
            "No trained model found. Run training first:\n\n"
            "```bash\npython src/train.py\n```"
        )
        return

    feature_cols = cfg["features"].get("columns", [])

    # ---- Input form --------------------------------------------------------
    st.subheader("Traffic Prediction Inputs")
    st.caption(
        "Only inputs actually used by the trained model are shown. "
        "The model uses: date/time features, lag features, rolling features, "
        "temperature, weather, holiday flag."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pred_date = st.date_input("Date", value=date.today(), key="pred_date")
        pred_time = st.time_input("Time (15-min interval)", value=time(8, 0), step=900, key="pred_time")
    with col_b:
        pred_temp = st.number_input(
            "Temperature (°C)", min_value=-5.0, max_value=50.0, value=25.0, step=0.5,
            help="Ambient temperature in degrees Celsius",
            key="pred_temp"
        )
        pred_weather = st.selectbox(
            "Weather", options=[0, 1, 2],
            format_func=lambda x: WEATHER_LABELS.get(x, str(x)),
            key="pred_weather"
        )
    with col_c:
        pred_is_holiday = st.checkbox("Public Holiday", value=False, key="pred_holiday")
        # Recent vehicle count for lag features
        prev_count = st.number_input(
            "Recent vehicle count (approx.)",
            min_value=0, max_value=2000, value=80,
            help="Approximate vehicle count from the most recent 15-min interval. "
                 "Used to compute lag features. Set to 0 if unknown.",
            key="pred_prev_count"
        )

    st.divider()

    # ---- Prediction --------------------------------------------------------
    if st.button("PREDICT TRAFFIC", type="primary", use_container_width=True):
        ts = pd.Timestamp(datetime.combine(pred_date, pred_time))

        # Build a minimal synthetic input row that matches training features.
        # We create 12 rows of history (max lag=8, max rolling=12) to avoid NaN warm-up.
        n_history = 12
        freq = "15min"
        hist_ts = pd.date_range(end=ts, periods=n_history + 1, freq=freq)

        # Linearly ramp up from ~prev_count*0.8 to prev_count
        hist_counts = np.linspace(prev_count * 0.8, prev_count, n_history + 1)

        # Use scaler to get the normalised scale of prev_count
        scaler_path = cfg["data"]["scaler_path"]
        try:
            scaler = load_scaler(scaler_path)
        except FileNotFoundError:
            st.error("Scaler not found. Please run training: `python src/train.py`")
            return

        # Build raw input DataFrame
        temp_vals = [pred_temp] * (n_history + 1)
        rows = pd.DataFrame({
            "timestamp": hist_ts,
            "location_id": "PRED",
            "vehicle_count": hist_counts,
            "is_holiday": [int(pred_is_holiday)] * (n_history + 1),
            "temperature": temp_vals,
            "weather_code": [pred_weather] * (n_history + 1),
        })

        # Normalise using training scaler
        scale_cols = ["vehicle_count", "temperature"]
        rows[scale_cols] = scaler.transform(rows[scale_cols])

        # Feature engineering
        rows = build_features(rows)
        if rows.empty:
            st.error("Feature engineering produced no rows. Try increasing 'Recent vehicle count'.")
            return

        # Take only the last row (prediction row)
        pred_row = rows.tail(1)[feature_cols]

        # Load model and predict
        try:
            model = load_model(cfg)
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            return

        pred_value = float(model.predict(pred_row)[0])
        traffic_label, traffic_colour = _classify_traffic(pred_value)

        # ---- Result card ---------------------------------------------------
        st.divider()
        st.subheader("Prediction Result")

        res_cols = st.columns([2, 1])
        with res_cols[0]:
            st.markdown("#### TRAFFIC FLOW PREDICTION")
            st.markdown(f"**Location:** {location_display}")
            st.markdown(f"**Date:** {pred_date.strftime('%d %B %Y')}")
            st.markdown(f"**Time:** {pred_time.strftime('%H:%M')}")
            st.markdown(f"**Weather:** {WEATHER_LABELS.get(pred_weather, str(pred_weather))}")
            st.markdown(f"**Holiday:** {'Yes' if pred_is_holiday else 'No'}")
            st.markdown(f"**Temperature:** {pred_temp}°C")

        with res_cols[1]:
            st.metric("Predicted Traffic Flow (normalised)", f"{pred_value:.4f}")
            st.markdown(
                f"<div style='background:{traffic_colour};color:white;padding:14px;"
                f"border-radius:8px;text-align:center;font-size:1.2em;font-weight:bold'>"
                f"Traffic Level: {traffic_label}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Note: Predictions are on the normalised (0-1) scale used during training. "
                "Thresholds: Low < 0.25, Moderate 0.25-0.50, High 0.50-0.75, Very High > 0.75."
            )

        # --- Chart 4: Actual vs Predicted (for this location's sensor if available) ---
        traffic_df = ctx.get("df")
        if traffic_df is not None and not traffic_df.empty:
            mapping_df = _load_mapping_df()
            loc_ids = get_location_ids_for_area(mapping_df, district=pred_district, sub_district=pred_sub)
            if loc_ids:
                sensor_df = traffic_df[traffic_df["location_id"].isin(loc_ids)].copy()
                if not sensor_df.empty:
                    st.subheader("Historical Traffic at Selected Location")
                    fig4, ax4 = plt.subplots(figsize=(14, 3.5))
                    ax4.plot(sensor_df["timestamp"], sensor_df["vehicle_count"],
                             linewidth=0.8, color="#2563EB", label="Historical data")
                    ax4.axhline(pred_value, color="#DC2626", linewidth=1.5,
                                linestyle="--", label=f"Prediction: {pred_value:.3f}")
                    ax4.set_title(f"Historical Traffic — {pred_district} / {pred_sub}", fontsize=11)
                    ax4.set_xlabel("Time")
                    ax4.set_ylabel("Vehicle Count (normalised)")
                    ax4.legend()
                    plt.xticks(rotation=25, ha="right", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)


# ---------------------------------------------------------------------------
# Tab 3 — Model Performance
# ---------------------------------------------------------------------------

def _tab_model_performance():
    st.header("Model Performance")

    cfg = _load_cfg()

    if not _model_is_trained(cfg):
        st.warning("No trained model found. Run `python src/train.py` first.")
        return

    # Metrics
    metrics = _load_metrics(cfg)
    if metrics:
        st.subheader("Evaluation Metrics (on held-out test set)")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics.get('mae', 'N/A'):.4f}", help="Mean Absolute Error")
        c2.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.4f}", help="Root Mean Squared Error")
        c3.metric("MAPE", f"{metrics.get('mape', 'N/A'):.2f} %", help="Mean Absolute Percentage Error")
        st.caption("All metrics computed on normalised (0-1) vehicle counts.")
    else:
        st.info("Metrics not found. Run training to generate them.")

    # Saved plots
    pred_plot = cfg.get("reports", {}).get("predictions_plot", "reports/prediction_vs_actual.png")
    res_plot  = cfg.get("reports", {}).get("residuals_plot",   "reports/residuals.png")

    if os.path.exists(pred_plot):
        st.subheader("Chart 5: Actual vs Predicted Traffic Flow")
        st.image(pred_plot, use_container_width=True)
    if os.path.exists(res_plot):
        st.subheader("Residuals Distribution")
        st.image(res_plot, use_container_width=True)

    if not os.path.exists(pred_plot) and not os.path.exists(res_plot):
        st.info("Evaluation plots are generated automatically during training.")

    # Feature columns
    feature_cols = cfg.get("features", {}).get("columns", [])
    if feature_cols:
        with st.expander(f"Feature columns used by the model ({len(feature_cols)})"):
            st.write(feature_cols)

    # Model type
    st.caption(f"Model type: **{cfg.get('model', {}).get('type', 'unknown')}**")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ctx = _sidebar()

    tab1, tab2, tab3 = st.tabs([
        "Data Explorer",
        "Predict Traffic",
        "Model Performance",
    ])

    with tab1:
        _tab_data_explorer(ctx)

    with tab2:
        _tab_predict(ctx)

    with tab3:
        _tab_model_performance()


if __name__ == "__main__":
    main()
