# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands

```bash
# Generate synthetic data (30 days, with geo columns)
python scripts/generate_sample_data.py --days 30

# Train model
python src/train.py
python src/train.py --model linear_regression   # override model

# Predict
python src/predict.py --input data/raw/sample_traffic.csv

# Dashboard
streamlit run src/dashboard.py

# Import official LGD village data (when available)
python scripts/update_locations.py --village data/lgd_villages.xlsx

# Validate location hierarchy
python scripts/update_locations.py --validate-only

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_locations.py -v
```

## Critical Platform Notes (Python 3.14 / pandas 3.x on Windows)

- **No `infer_datetime_format`** — removed in pandas 2.0. Use bare `pd.to_datetime(col)`.
- **No `fillna(method=...)`** — removed in pandas 3.0. Use `.ffill()` / `.bfill()` directly.
- **`groupby().apply()` strips the groupby key column** in pandas 3.x. Use `.groupby()[col].transform()` or `.groupby()[col].shift()` instead.
- **No Unicode in print() on Windows cp1252** — avoid `->`, `<-` arrows as Unicode; use ASCII.
- **`select_dtypes(include="object")`** emits Pandas4Warning — use `include=["object", "str"]`.

## Location Data Architecture

- **`data/locations.csv`** — authoritative Karnataka hierarchy (LGD-verified). 31 districts, 192 taluks. Villages empty until official LGD export is imported.
- **`data/location_id_mapping.csv`** — maps LOC_01..LOC_05 to Karnataka districts/taluks. DEMO assignments only; not real sensor placements.
- **`src/location_data.py`** — all hierarchy queries (`get_districts`, `get_sub_districts`, `get_villages`, `validate_hierarchy_combination`). Always use this module; never hard-code district lists.
- **`scripts/update_locations.py`** — LGD import tool. Handles CSV/XLS district, sub-district, and village exports from https://lgdirectory.gov.in/

## Model Feature Columns — Geo Columns Must Never Enter the Model

The traffic CSV now contains `country`, `state`, `district`, `sub_district` string columns.
[`get_feature_columns()`](src/feature_engineering.py) explicitly excludes all non-numeric and geo-metadata columns. If you add new geo columns to the CSV, add them to the `meta_exclude` set in `get_feature_columns()` or they will break model training.

## Pluggable Model System

All models implement [`TrafficModel`](src/model_interface.py) abstract base class. To add LSTM/GRU:
1. Create `src/model_lstm.py` implementing `TrafficModel`.
2. Register in `MODEL_REGISTRY` in [`src/model_baseline.py`](src/model_baseline.py).
3. Set `model.type: lstm` in `config.yaml`. No other changes needed.

## Feature Engineering Constraints

- Lag and rolling features computed **per `location_id`** using `.groupby()[col].shift()` / `.transform()`.
- Feature column list auto-written to `config.yaml → features.columns`. Never edit manually.
- Warm-up rows (first `max(lags)` per location) dropped by `build_features()`.

## Normalisation

- `vehicle_count` + `temperature` Min-Max scaled at train time. Scaler saved to `models/scaler.pkl`.
- Inference always loads training scaler (`fit=False`). Predictions are on 0-1 scale.
- MAPE uses `max(|y_true|, 0.01)` denominator guard.

## Time-Based Split

Train/test split is by **global timestamp** (last `test_split` fraction of timestamps = test set). Never use `train_test_split(shuffle=True)`.

## Data Source Honesty Rule

Never claim villages are loaded unless `data/locations.csv` actually contains village records. The seed file has 0 villages. `location_summary()` reports the true count.

## Stack

Python 3.14 · pandas 3.x · scikit-learn 1.9 · matplotlib · Streamlit · joblib · pytest
