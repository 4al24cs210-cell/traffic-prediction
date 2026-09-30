# Traffic Flow Prediction Model — Karnataka

A modular, end-to-end Python machine learning system for predicting urban traffic flow using historical sensor data.

---

## Features

- Ingests traffic data from CSV files
- **Karnataka geographic hierarchy** — India > Karnataka > District > Taluk > Village, powered by LGD-verified data
- Hierarchical dependent dropdowns and location-aware filtering in the dashboard
- Automated data cleaning, normalisation, and feature engineering
- Pluggable model architecture (Linear Regression, Random Forest — LSTM/GRU ready)
- Evaluation with MAE, RMSE, and MAPE
- Interactive Streamlit dashboard: Data Explorer, Predict Traffic, Model Performance

---

## Requirements

- Python 3.10 or higher
- All dependencies listed in `requirements.txt`

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — Generate synthetic training data (skip if you have real data)
python scripts/generate_sample_data.py

# Step 2 — Train the model
python src/train.py

# Step 3 — Run predictions on existing data
python src/predict.py --input data/raw/sample_traffic.csv

# Step 4 — Open the dashboard
streamlit run src/dashboard.py
```

---

## Command Reference

### Generate sample data

```bash
python scripts/generate_sample_data.py
python scripts/generate_sample_data.py --output data/raw/custom.csv --days 60 --seed 0
```

### Train

```bash
python src/train.py                             # uses config.yaml defaults
python src/train.py --model linear_regression   # override model type
python src/train.py --config path/to/config.yaml
```

### Predict

```bash
python src/predict.py --input data/raw/sample_traffic.csv
python src/predict.py --input data/raw/new_data.csv --output results/predictions.csv
```

### Dashboard

```bash
streamlit run src/dashboard.py
```

### Run tests

```bash
pytest tests/
pytest tests/test_data_ingestion.py   # single test file
pytest tests/ -v                      # verbose
```

---

## CSV Format

Every input CSV **must** contain these columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime string | ISO 8601 or common locale format (`YYYY-MM-DD HH:MM:SS`) |
| `location_id` | string | Sensor or intersection identifier |
| `vehicle_count` | integer | Number of vehicles in the interval |

### Optional columns (used as features if present)

| Column | Type | Description |
|---|---|---|
| `day_of_week` | integer (0–6) | Pre-computed weekday |
| `is_holiday` | integer (0/1) | Public holiday flag |
| `temperature` | float | Ambient temperature in °C |
| `weather_code` | integer | 0=clear, 1=rain, 2=fog |

### Example row

```
timestamp,location_id,vehicle_count,is_holiday,temperature,weather_code
2023-01-02 07:30:00,LOC_01,87,0,14.2,0
```

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `data.raw_path` | `data/raw/sample_traffic.csv` | Input CSV path |
| `data.processed_path` | `data/processed/features.csv` | Feature matrix output path |
| `data.scaler_path` | `models/scaler.pkl` | Saved MinMaxScaler |
| `model.type` | `random_forest` | Model to use: `linear_regression` or `random_forest` |
| `model.save_path` | `models/model.pkl` | Trained model output path |
| `training.test_split` | `0.2` | Fraction of data reserved for testing (time-based) |
| `training.random_seed` | `42` | Random seed for reproducibility |
| `random_forest.n_estimators` | `100` | Number of trees |
| `random_forest.max_depth` | `10` | Maximum tree depth |
| `reports.metrics_path` | `reports/metrics.json` | Evaluation metrics output |
| `reports.predictions_plot` | `reports/prediction_vs_actual.png` | Predictions chart |
| `reports.residuals_plot` | `reports/residuals.png` | Residuals histogram |

> **Note:** `features.columns` is written automatically by `train.py`. Do not edit it manually.

---

## Project Structure

```
TrafficFlowPrediction/
├── data/
│   ├── raw/                          # Input traffic CSV files
│   ├── processed/                    # Feature-engineered data
│   ├── locations.csv                 # Karnataka LGD hierarchy (districts + taluks)
│   ├── karnataka_districts_lgd.csv   # District-level LGD reference
│   └── location_id_mapping.csv       # Mapping: LOC_01..LOC_05 -> geographic area
├── models/                           # Saved model (.pkl) and scaler (.pkl)
├── reports/                          # Evaluation plots and metrics.json
├── scripts/
│   ├── generate_sample_data.py       # Synthetic data generator
│   └── update_locations.py           # LGD data import/update tool
├── src/
│   ├── data_ingestion.py             # CSV loading and validation
│   ├── preprocessing.py              # Cleaning and normalisation
│   ├── feature_engineering.py        # Lag, rolling, and time features
│   ├── model_interface.py            # Abstract TrafficModel base class
│   ├── model_baseline.py             # LinearRegression and RandomForest
│   ├── location_data.py              # Location hierarchy loader and queries
│   ├── train.py                      # Training entry point (CLI)
│   ├── evaluate.py                   # Metrics computation and plots
│   ├── predict.py                    # Inference entry point (CLI)
│   └── dashboard.py                  # Streamlit application
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_predict.py
│   └── test_locations.py             # Karnataka hierarchy + validation tests
├── config.yaml                       # All configuration
├── requirements.txt
└── README.md
```

---

## Adding a New Model (LSTM/GRU)

The pluggable model interface makes adding advanced models straightforward:

1. Create `src/model_lstm.py`.
2. Define `class LSTMModel(TrafficModel)` and implement `fit`, `predict`, `save`, `load`.
3. Register it in `src/model_baseline.py` → `MODEL_REGISTRY`:
   ```python
   from src.model_lstm import LSTMModel
   MODEL_REGISTRY["lstm"] = LSTMModel
   ```
4. Set `model.type: lstm` in `config.yaml`.

No changes to `train.py`, `predict.py`, or the dashboard are required.

---

## Karnataka Location Hierarchy (LGD Data)

### Official Data Source

> Government of India — **Local Government Directory (LGD)**
> URL: https://lgdirectory.gov.in/
> Karnataka State LGD Code: **29**

### Data Currently Loaded

| Level | Count | Source |
|---|---|---|
| Districts | **31** | LGD-verified (all current Karnataka districts) |
| Sub-districts (Taluks) | **192** | LGD-verified taluk names |
| Villages | **0** | Requires official LGD village export (see below) |

> **Note:** Geographic administrative data (districts, taluks, villages) and synthetic traffic
> observations are **two separate datasets**. The traffic CSV contains simulated sensor readings;
> the location hierarchy provides the administrative context for filtering and display only.

### Importing the Full LGD Dataset (Villages)

To load the complete ~30,764 village records for Karnataka:

1. Visit https://lgdirectory.gov.in/
2. Navigate to: Reports > Village > Download All Villages (Karnataka)
3. Download the XLS/CSV export
4. Run the import script:

```bash
python scripts/update_locations.py \
    --district data/lgd_districts.xlsx \
    --subdistrict data/lgd_subdistricts.xlsx \
    --village data/lgd_villages.xlsx
```

5. Validate the result:

```bash
python scripts/update_locations.py --validate-only
```

### Location ID Mapping

The synthetic traffic data uses internal sensor IDs (`LOC_01`–`LOC_05`).
These are mapped to Karnataka locations in `data/location_id_mapping.csv`.

> **These mappings are illustrative demo assignments only.**
> They are NOT real sensor placements. The `notes` column in the mapping
> file explicitly marks them as `DEMO`.

---

## Limitations and Known Issues

| Limitation | Detail |
|---|---|
| Single-step prediction | The model predicts one interval ahead. Multi-step forecasting is not yet implemented. |
| Cold-start warm-up | The first `max(lags) + max(windows)` rows per location are dropped as NaN warm-up during inference. |
| Normalised output | Predictions are in normalised (0-1) scale. Inverse-transform using `models/scaler.pkl` if raw vehicle counts are needed. |
| CPU only | Default setup runs on CPU. GPU support can be added for PyTorch-based models. |
| No real-time ingestion | Batch CSV only; streaming is not supported. |
| Villages not loaded | Village-level data requires a manual LGD export. The application does NOT claim complete village coverage until the official LGD dataset is imported via `scripts/update_locations.py`. |
| Demo sensor mapping | LOC_01–LOC_05 geographic assignments are illustrative only, not real sensor placements. |
| LGD data currency | LGD district and taluk names are correct as of the bundled seed file date. Run `scripts/update_locations.py` after any LGD boundary/name changes. |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests are located in `tests/` and cover data ingestion, preprocessing, feature engineering, models, and the end-to-end prediction pipeline.
