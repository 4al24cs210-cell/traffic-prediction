"""
predict.py
----------
Load a saved model and run inference on new traffic data.

The preprocessing and feature engineering applied during inference must
exactly match training.  The saved scaler (models/scaler.pkl) and the
feature column list (config.yaml → features.columns) guarantee this.

Known limitation
----------------
Lag and rolling features require prior history.  When a CSV is submitted
cold (no preceding rows), the first max(lags) + max(windows) rows per
location will be dropped as warm-up NaN rows.  Document this to users.

Usage
-----
    python src/predict.py --input data/raw/sample_traffic.csv
    python src/predict.py --input new_data.csv --output predictions.csv
    python src/predict.py --input new_data.csv --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import load_data
from src.preprocessing import fill_missing, cap_outliers, load_scaler
from src.feature_engineering import build_features, get_feature_columns
from src.model_baseline import build_model, LinearRegressionModel, RandomForestModel
from src.model_interface import TrafficModel


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(cfg: dict) -> TrafficModel:
    """Instantiate and load a saved model based on config.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml.

    Returns
    -------
    TrafficModel
    """
    model = build_model(cfg)
    model.load(cfg["model"]["save_path"])
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model: TrafficModel,
    scaler_path: str,
    feature_cols: list[str],
    target_col: str = "vehicle_count",
) -> pd.DataFrame:
    """Apply preprocessing + feature engineering, then predict.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or lightly-cleaned traffic DataFrame (must have required columns).
    model : TrafficModel
        Fitted model (already loaded from disk).
    scaler_path : str
        Path to the saved MinMaxScaler (from training).
    feature_cols : list[str]
        Ordered list of feature columns to pass to the model.
        Must match exactly what was used at training time.
    target_col : str

    Returns
    -------
    pd.DataFrame
        Original DataFrame rows (after warm-up drop) with an appended
        ``predicted_vehicle_count`` column.
    """
    # 1. Clean
    df = fill_missing(df)
    df = cap_outliers(df, col=target_col)

    # 2. Normalise with the training scaler (fit=False → never refit)
    scale_cols = [target_col]
    for col in ["temperature"]:
        if col in df.columns:
            scale_cols.append(col)
    scaler = load_scaler(scaler_path)
    df[scale_cols] = scaler.transform(df[scale_cols])

    # 3. Feature engineering
    df = build_features(df)

    # 4. Ensure all required feature columns are present
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature columns missing after engineering: {missing_cols}. "
            "Ensure the input CSV covers the same columns as the training data."
        )

    X = df[feature_cols]

    # 5. Predict
    preds = model.predict(X)
    df = df.copy()
    df["predicted_vehicle_count"] = preds

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def predict(
    input_path: str,
    output_path: str | None = None,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Full prediction pipeline: load → preprocess → infer → (optionally) save.

    Parameters
    ----------
    input_path : str
        Path to input CSV.
    output_path : str, optional
        Where to write the result CSV.  If None, prints a head summary.
    config_path : str

    Returns
    -------
    pd.DataFrame
        Result DataFrame with ``predicted_vehicle_count`` column.
    """
    cfg = _load_config(config_path)

    feature_cols = cfg["features"].get("columns", [])
    if not feature_cols:
        raise RuntimeError(
            "No feature columns found in config.yaml. "
            "Run training first (python src/train.py)."
        )

    print(f"[predict] Loading data: {input_path}")
    df = load_data(input_path)

    print(f"[predict] Loading model: {cfg['model']['save_path']}")
    model = load_model(cfg)

    result = run_inference(
        df=df,
        model=model,
        scaler_path=cfg["data"]["scaler_path"],
        feature_cols=feature_cols,
        target_col=cfg["features"].get("target", "vehicle_count"),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"[predict] Predictions saved -> {output_path}")
    else:
        print("\n[predict] Sample predictions (first 10 rows):")
        show_cols = [c for c in ["timestamp", "location_id", "vehicle_count", "predicted_vehicle_count"] if c in result.columns]
        print(result[show_cols].head(10).to_string(index=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic flow inference.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path (optional)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()
    predict(input_path=args.input, output_path=args.output, config_path=args.config)


if __name__ == "__main__":
    main()
