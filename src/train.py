"""
train.py
--------
End-to-end training entry point for the Traffic Flow Prediction Model.

Usage
-----
    python src/train.py
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --model random_forest

What this script does
---------------------
1. Load and validate the raw CSV.
2. Clean / preprocess (fill missing, cap outliers, normalise).
3. Engineer features (time, lag, rolling).
4. Split into train / test by time (no shuffling).
5. Train the model selected in config.yaml (or --model flag).
6. Evaluate on the test set (MAE, RMSE, MAPE).
7. Save the model and scaler to disk.
8. Save evaluation plots and metrics JSON.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import load_data, data_quality_report
from src.preprocessing import fill_missing, cap_outliers, normalise
from src.feature_engineering import build_features, get_feature_columns
from src.model_baseline import build_model
from src.evaluate import run_evaluation


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False)


# ---------------------------------------------------------------------------
# Train/test split (time-based)
# ---------------------------------------------------------------------------

def time_based_split(df, test_fraction: float = 0.2):
    """Split *df* by time: last *test_fraction* rows become the test set.

    The split point is the same for every location, based on the global
    timestamp order (all timestamps are sorted before splitting).

    Returns
    -------
    train_df, test_df : pd.DataFrame, pd.DataFrame
    """
    # Use global timestamp ordering, not per-location
    all_timestamps = df["timestamp"].sort_values().unique()
    n_test = int(len(all_timestamps) * test_fraction)
    split_ts = all_timestamps[-n_test]

    train_df = df[df["timestamp"] < split_ts].copy()
    test_df = df[df["timestamp"] >= split_ts].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(config_path: str = "config.yaml", model_override: str | None = None) -> dict:
    """Run the full training pipeline.

    Parameters
    ----------
    config_path : str
    model_override : str, optional
        Override the model type in config (e.g. ``"linear_regression"``).

    Returns
    -------
    dict
        Evaluation metrics (mae, rmse, mape).
    """
    cfg = load_config(config_path)

    if model_override:
        cfg["model"]["type"] = model_override

    print(f"\n{'='*55}")
    print("  TRAFFIC FLOW PREDICTION — TRAINING")
    print(f"  Model : {cfg['model']['type']}")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    raw_path = cfg["data"]["raw_path"]
    print(f"[train] Loading data from: {raw_path}")
    df = load_data(raw_path)
    data_quality_report(df)

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    print("\n[train] Preprocessing…")
    df = fill_missing(df)
    df = cap_outliers(df, col="vehicle_count")

    # Columns to normalise (vehicle_count + optional numeric features)
    scale_cols = ["vehicle_count"]
    for col in ["temperature"]:
        if col in df.columns:
            scale_cols.append(col)

    scaler_path = cfg["data"]["scaler_path"]
    df = normalise(df, cols=scale_cols, scaler_path=scaler_path, fit=True)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("[train] Engineering features…")
    df = build_features(df)

    feature_cols = get_feature_columns(df)
    target_col = cfg["features"].get("target", "vehicle_count")

    print(f"[train] Features ({len(feature_cols)}): {feature_cols}")

    # Persist feature column list so inference uses exactly the same columns
    cfg["features"]["columns"] = feature_cols
    save_config(cfg, config_path)

    # ------------------------------------------------------------------
    # 4. Time-based split
    # ------------------------------------------------------------------
    test_frac = cfg["training"].get("test_split", 0.2)
    train_df, test_df = time_based_split(df, test_fraction=test_frac)

    print(f"\n[train] Dataset split (time-based, test={test_frac:.0%}):")
    print(f"        Train rows : {len(train_df):,}  | from {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"        Test rows  : {len(test_df):,}  | from {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ------------------------------------------------------------------
    # 5. Model training
    # ------------------------------------------------------------------
    print(f"\n[train] Training {cfg['model']['type']}…")
    model = build_model(cfg)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("[train] Evaluating on test set…")
    y_pred = model.predict(X_test)
    timestamps = test_df["timestamp"].values
    metrics = run_evaluation(y_pred=y_pred, y_true=y_test.values, timestamps=timestamps, cfg=cfg)

    # ------------------------------------------------------------------
    # 7. Save processed features & model
    # ------------------------------------------------------------------
    processed_path = cfg["data"]["processed_path"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[train] Processed features saved -> {processed_path}")

    model.save(cfg["model"]["save_path"])

    print("\n[train] Training complete.\n")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Traffic Flow Prediction Model.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["linear_regression", "random_forest"],
        help="Override the model type in config.yaml.",
    )
    args = parser.parse_args()
    train(config_path=args.config, model_override=args.model)


if __name__ == "__main__":
    main()
