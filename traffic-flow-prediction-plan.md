# Traffic Flow Prediction Model — Development Plan

## Overview

Build a modular, end-to-end Python machine learning system that:
- Accepts historical traffic data in CSV format.
- Preprocesses data and engineers time-series features.
- Trains a machine learning model to predict future traffic flow (vehicle count per time interval).
- Evaluates the model using MAE, RMSE, and MAPE.
- Displays results in a Streamlit dashboard showing raw data, predictions, and model metrics.
- Is structured so that advanced models (LSTM, GRU) can be added later without restructuring.

**Scope:** Local batch prediction only. No streaming, no authentication, no deployment.

---

## Assumptions

1. Traffic flow = vehicle count per 15-minute time interval per location.
2. Input CSV has at minimum three columns: `timestamp`, `location_id`, `vehicle_count`.
3. Additional columns (weather, road type, speed) are treated as optional features.
4. No real dataset is available initially; a synthetic sample CSV will be generated.
5. Predictions are single-step ahead (next interval) unless extended later.
6. All computation runs on a standard CPU laptop; GPU support is optional.
7. Model artifacts are saved to disk using `.pkl` (scikit-learn) or `.pt` (PyTorch).
8. The dashboard is local-only (no public hosting required).

---

## Project Structure

```
TrafficFlowPrediction/
├── data/
│   ├── raw/                  # Original CSV files (input)
│   └── processed/            # Cleaned, feature-engineered data
├── models/                   # Saved model artifacts
├── reports/                  # Evaluation plots and metric summaries
├── src/
│   ├── data_ingestion.py     # Load and validate CSV
│   ├── preprocessing.py      # Clean, normalise, handle missing values
│   ├── feature_engineering.py# Time features, lag features, rolling stats
│   ├── model_baseline.py     # Linear Regression / Random Forest
│   ├── model_interface.py    # Abstract base class for all models
│   ├── train.py              # Training entry point (CLI)
│   ├── evaluate.py           # Compute MAE, RMSE, MAPE; save report
│   ├── predict.py            # Load saved model, run inference
│   └── dashboard.py          # Streamlit app
├── scripts/
│   └── generate_sample_data.py  # Create synthetic CSV for development
├── config.yaml               # Hyperparameters and file paths
├── requirements.txt
└── README.md
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding and Sample Data

**Status:** `[ ] pending`

**Intent:**  
Create the directory structure, configuration file, dependency list, and a script that generates a realistic synthetic CSV dataset so that all subsequent tasks can be developed and tested without a real dataset.

**Expected Outcomes:**
- All directories under the structure above exist.
- `requirements.txt` lists all needed packages.
- `config.yaml` defines file paths and default hyperparameters.
- Running `python scripts/generate_sample_data.py` produces `data/raw/sample_traffic.csv` with ≥ 10,000 rows covering multiple locations and timestamps.
- CSV has columns: `timestamp`, `location_id`, `vehicle_count`, `day_of_week`, `is_holiday`.

**Todo List:**
1. Create directory tree.
2. Write `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, pyyaml, joblib).
3. Write `config.yaml` with keys: `data.raw_path`, `data.processed_path`, `model.type`, `model.save_path`, `training.test_split`, `training.random_seed`.
4. Write `scripts/generate_sample_data.py` using NumPy to simulate daily traffic patterns with noise, weekday/weekend variation, and a holiday flag.
5. Write `README.md` with setup instructions and command reference.

**Relevant Context:**
- Config keys used by all subsequent modules — agree on naming before writing other modules.
- Sample data must realistically mirror what a real CSV would look like (same column names, same data types).

---

### Sub-Task 2 — Data Ingestion and Validation

**Status:** `[ ] pending`

**Intent:**  
Build the module that loads a CSV file, validates its schema, and reports data quality issues (missing values, wrong types, duplicate rows) before any processing occurs.

**Expected Outcomes:**
- `src/data_ingestion.py` exposes a `load_data(path: str) -> pd.DataFrame` function.
- Function raises a descriptive `ValueError` if required columns are missing.
- Prints a summary: row count, date range, locations found, missing-value counts per column.
- Works correctly on the synthetic CSV from Sub-Task 1.

**Todo List:**
1. Implement `load_data()` — reads CSV, parses `timestamp` as datetime, validates required columns.
2. Implement `validate_schema()` — checks column names and data types.
3. Implement `data_quality_report()` — prints missing value counts, duplicate row count, date range.
4. Write unit tests (3–5 assertions using `pytest`) in `tests/test_data_ingestion.py`.

**Relevant Context:**
- `load_data` is the entry point for both `train.py` and `predict.py`.
- Timestamp parsing must be flexible (ISO 8601 and common locale formats).

---

### Sub-Task 3 — Preprocessing and Feature Engineering

**Status:** `[ ] pending`

**Intent:**  
Transform raw data into a model-ready feature matrix. This is the most analytically important module — good features directly determine model quality.

**Expected Outcomes:**
- `src/preprocessing.py` handles missing-value imputation, outlier capping, and normalisation.
- `src/feature_engineering.py` generates: hour-of-day, day-of-week, month, is_weekend, lag features (t-1, t-2, t-4, t-8 intervals), rolling mean (1h, 3h windows).
- Output is a clean `pd.DataFrame` with a defined feature list saved to `data/processed/`.
- Scaler is saved to `models/scaler.pkl` so inference uses the same scale as training.

**Todo List:**
1. Implement `fill_missing(df)` — forward-fill then backward-fill gaps ≤ 3 steps; flag longer gaps.
2. Implement `cap_outliers(df, col, z_thresh=3.0)` — Z-score capping for `vehicle_count`.
3. Implement `normalise(df, cols)` — Min-Max scaling using `sklearn.preprocessing.MinMaxScaler`; save scaler.
4. Implement `add_time_features(df)` — extract hour, day, month, is_weekend from timestamp.
5. Implement `add_lag_features(df, lags=[1,2,4,8])` — shift `vehicle_count` by lag intervals per location.
6. Implement `add_rolling_features(df, windows=[4,12])` — rolling mean over 4 and 12 intervals (1h, 3h).
7. Write `tests/test_preprocessing.py` covering edge cases: all-NaN column, single-row input.

**Relevant Context:**
- Lag and rolling features must be computed per `location_id` group, not across the whole dataset.
- The feature list produced here must be documented in `config.yaml` under `features.columns` so the model and inference pipeline use exactly the same columns.

---

### Sub-Task 4 — Model Interface and Baseline Models

**Status:** `[ ] pending`

**Intent:**  
Define an abstract `TrafficModel` base class that all models (baseline and future deep-learning) must implement. Then build two baseline models: Linear Regression and Random Forest. This enforces the pluggable architecture.

**Expected Outcomes:**
- `src/model_interface.py` defines `TrafficModel` with abstract methods: `fit(X, y)`, `predict(X)`, `save(path)`, `load(path)`.
- `src/model_baseline.py` implements `LinearRegressionModel` and `RandomForestModel` using this interface.
- Both models can be trained and saved to disk.
- Swapping models requires only a config change, not code changes.

**Todo List:**
1. Write `TrafficModel` abstract base class in `src/model_interface.py`.
2. Implement `LinearRegressionModel` wrapping `sklearn.linear_model.LinearRegression`.
3. Implement `RandomForestModel` wrapping `sklearn.ensemble.RandomForestRegressor` with configurable `n_estimators` and `max_depth` from `config.yaml`.
4. Implement `save()` / `load()` using `joblib` for both classes.
5. Write `tests/test_models.py` — fit on 50 synthetic rows, assert predictions are numeric and shape-correct.

**Relevant Context:**
- Future `LSTMModel` and `GRUModel` will inherit `TrafficModel` — keep the interface minimal and generic.
- `config.yaml` key `model.type` selects which class to instantiate in `train.py`.

---

### Sub-Task 5 — Training Pipeline

**Status:** `[ ] pending`

**Intent:**  
Build the end-to-end training entry point that reads config, loads and preprocesses data, trains the selected model, and saves the artifact — runnable from the command line with one command.

**Expected Outcomes:**
- Running `python src/train.py --config config.yaml` completes without error on the synthetic dataset.
- Prints: dataset shape, train/test split sizes, model type, and training completion message.
- Saves trained model to `models/` path defined in config.
- Saves processed feature data to `data/processed/`.

**Todo List:**
1. Implement `train.py` CLI entry point using `argparse`.
2. Wire together: `load_data` → `preprocessing` → `feature_engineering` → `train/test split` → `model.fit()` → `model.save()`.
3. Train/test split by time (last N% of data = test set) — **not** random shuffle, to respect temporal order.
4. Log split boundary date and sizes to console.
5. Optionally accept `--model` flag to override `config.yaml` model type.

**Relevant Context:**
- Time-based splitting (not random) is critical for time-series validity. Random splitting would leak future data into training.
- Feature column list must be read from `config.yaml` `features.columns` to guarantee consistency with inference.

---

### Sub-Task 6 — Evaluation Module

**Status:** `[ ] pending`

**Intent:**  
Compute and report model performance metrics and produce visualisation plots so the model quality can be assessed objectively.

**Expected Outcomes:**
- `src/evaluate.py` computes MAE, RMSE, and MAPE on the held-out test set.
- Saves a `reports/metrics.json` file with the numeric results.
- Saves a `reports/prediction_vs_actual.png` plot (line chart: actual vs predicted over time).
- Saves a `reports/residuals.png` plot (residual distribution histogram).
- Prints a formatted metric summary to console.

**Todo List:**
1. Implement `compute_metrics(y_true, y_pred) -> dict` returning MAE, RMSE, MAPE.
2. Implement `plot_predictions(y_true, y_pred, timestamps)` — line chart saved to `reports/`.
3. Implement `plot_residuals(y_true, y_pred)` — histogram saved to `reports/`.
4. Implement `save_metrics(metrics: dict, path: str)` — write JSON.
5. Integrate evaluation call into the end of `train.py`.

**Relevant Context:**
- MAPE is undefined when `y_true = 0`; use a small epsilon guard (e.g., `max(abs(y_true), 1e-8)`).
- Plots must be saved (not shown interactively) so they work in headless/server environments.

---

### Sub-Task 7 — Inference / Prediction Module

**Status:** `[ ] pending`

**Intent:**  
Build a standalone prediction module that loads a saved model and scaler, accepts new CSV input, and outputs predicted traffic flow values — enabling the dashboard and future API integration.

**Expected Outcomes:**
- `src/predict.py` loads model from `models/` and scaler from `models/scaler.pkl`.
- Accepts a CSV path as input, runs the same feature engineering pipeline, and returns a DataFrame with a `predicted_vehicle_count` column.
- Can be run from CLI: `python src/predict.py --input data/raw/new_data.csv --output results.csv`.

**Todo List:**
1. Implement `load_model(config)` — reads model type from config, loads correct artifact.
2. Implement `run_inference(df, model, scaler, feature_cols) -> pd.DataFrame`.
3. Ensure feature engineering in inference matches training exactly (same lag/rolling logic, same scaler).
4. Write CLI wrapper with `argparse`.
5. Write `tests/test_predict.py` — assert output DataFrame has correct shape and no NaN predictions.

**Relevant Context:**
- The scaler must be the one saved during training — never refit on inference data.
- Lag features require recent history; if the input CSV starts cold (no prior rows), the first N rows will have NaN lags — document this limitation clearly.

---

### Sub-Task 8 — Streamlit Dashboard

**Status:** `[ ] pending`

**Intent:**  
Build a simple interactive dashboard for visualising raw traffic data, model predictions, and evaluation metrics — making results accessible without command-line knowledge.

**Expected Outcomes:**
- Running `streamlit run src/dashboard.py` opens a browser page with no errors.
- Dashboard has three sections: **Data Explorer**, **Predictions**, **Model Performance**.
- User can select location and date range from sidebar widgets.
- Prediction vs actual chart updates based on selection.
- Metric cards display MAE, RMSE, MAPE loaded from `reports/metrics.json`.
- Works offline; no external API calls.

**Todo List:**
1. Implement sidebar: file uploader (CSV), location selector, date range picker.
2. Implement **Data Explorer** tab: line chart of raw `vehicle_count` over time.
3. Implement **Predictions** tab: overlay chart of actual vs predicted; calls `run_inference()` on uploaded data.
4. Implement **Model Performance** tab: display metric values from `reports/metrics.json`; embed saved plots.
5. Handle graceful state when no model is trained yet (display informative message).

**Relevant Context:**
- Streamlit re-runs the entire script on each widget interaction — use `@st.cache_data` on data loading and inference to avoid redundant computation.
- The dashboard imports directly from `src/` modules — no separate API layer is needed.

---

### Sub-Task 9 — README and Final Documentation

**Status:** `[ ] pending`

**Intent:**  
Produce a complete README that lets a new user set up the project, run training, and open the dashboard with copy-paste commands.

**Expected Outcomes:**
- `README.md` covers: project description, setup (virtualenv + pip install), quick-start commands, CSV format specification, config reference, and known limitations.
- All commands in the README are tested and work.

**Todo List:**
1. Write **Setup** section: Python version requirement, virtualenv creation, `pip install -r requirements.txt`.
2. Write **Quick Start** section: generate sample data → train → evaluate → open dashboard.
3. Write **CSV Format** section: required columns, data types, example row.
4. Write **Config Reference** section: table of all `config.yaml` keys with descriptions and defaults.
5. Write **Limitations / Future Work** section: single-step prediction only, LSTM/GRU extension path, real dataset integration.

---

## Technology Stack Summary

| Layer | Library / Tool |
|---|---|
| Data handling | pandas, NumPy |
| ML models | scikit-learn |
| Deep learning (future) | PyTorch (optional) |
| Serialisation | joblib |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Config | PyYAML |
| Testing | pytest |
| Python version | 3.10+ |

---

## Execution Order

Sub-tasks are designed to be implemented in order, each building on the previous:

```
1 (Scaffold) → 2 (Ingest) → 3 (Preprocess) → 4 (Models) → 5 (Train)
                                                           → 6 (Evaluate)
                                                           → 7 (Predict)
                                                           → 8 (Dashboard)
                                                           → 9 (README)
```

Sub-tasks 6, 7, and 8 can proceed in parallel once Sub-task 5 is complete.
