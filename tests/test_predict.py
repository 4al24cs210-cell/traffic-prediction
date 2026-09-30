"""
tests/test_predict.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_sample_data import generate
from src.train import train, load_config
from src.predict import predict, run_inference, load_model
from src.preprocessing import load_scaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_project(tmp_path_factory):
    """Generate sample data, write a minimal config, and train a model.
    Returns the tmp directory path and config path."""
    root = tmp_path_factory.mktemp("project")

    # Generate tiny dataset (5 days to keep test fast)
    data_path = str(root / "data" / "raw" / "traffic.csv")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    generate(output_path=data_path, days=10, seed=0)

    # Write config pointing to tmp dirs
    import yaml
    cfg = {
        "data": {
            "raw_path": data_path,
            "processed_path": str(root / "data" / "processed" / "features.csv"),
            "scaler_path": str(root / "models" / "scaler.pkl"),
        },
        "model": {
            "type": "random_forest",
            "save_path": str(root / "models" / "model.pkl"),
        },
        "training": {"test_split": 0.2, "random_seed": 42},
        "random_forest": {"n_estimators": 5, "max_depth": 3},
        "features": {"target": "vehicle_count", "columns": []},
        "reports": {
            "metrics_path": str(root / "reports" / "metrics.json"),
            "predictions_plot": str(root / "reports" / "pred.png"),
            "residuals_plot": str(root / "reports" / "res.png"),
        },
    }
    config_path = str(root / "config.yaml")
    with open(config_path, "w") as fh:
        yaml.safe_dump(cfg, fh)

    os.makedirs(str(root / "models"), exist_ok=True)
    os.makedirs(str(root / "reports"), exist_ok=True)
    os.makedirs(str(root / "data" / "processed"), exist_ok=True)

    metrics = train(config_path=config_path)
    return {"root": root, "config_path": config_path, "data_path": data_path, "metrics": metrics}


# ---------------------------------------------------------------------------
# Test training side effects
# ---------------------------------------------------------------------------

def test_training_produces_model_file(trained_project):
    cfg = load_config(trained_project["config_path"])
    assert os.path.exists(cfg["model"]["save_path"])


def test_training_produces_scaler_file(trained_project):
    cfg = load_config(trained_project["config_path"])
    assert os.path.exists(cfg["data"]["scaler_path"])


def test_training_produces_metrics_json(trained_project):
    cfg = load_config(trained_project["config_path"])
    assert os.path.exists(cfg["reports"]["metrics_path"])


def test_training_metrics_keys(trained_project):
    metrics = trained_project["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics


# ---------------------------------------------------------------------------
# Test prediction pipeline
# ---------------------------------------------------------------------------

def test_predict_returns_dataframe(trained_project):
    result = predict(
        input_path=trained_project["data_path"],
        output_path=None,
        config_path=trained_project["config_path"],
    )
    assert isinstance(result, pd.DataFrame)


def test_predict_has_prediction_column(trained_project):
    result = predict(
        input_path=trained_project["data_path"],
        config_path=trained_project["config_path"],
    )
    assert "predicted_vehicle_count" in result.columns


def test_predict_no_nan_predictions(trained_project):
    result = predict(
        input_path=trained_project["data_path"],
        config_path=trained_project["config_path"],
    )
    assert not result["predicted_vehicle_count"].isna().any()


def test_predict_output_shape_matches_input(trained_project):
    """Result must have at most as many rows as input (warm-up may drop some)."""
    import pandas as pd
    raw = pd.read_csv(trained_project["data_path"])
    result = predict(
        input_path=trained_project["data_path"],
        config_path=trained_project["config_path"],
    )
    assert len(result) <= len(raw)
    assert len(result) > 0


def test_predict_saves_output_csv(trained_project, tmp_path):
    out = str(tmp_path / "out.csv")
    predict(
        input_path=trained_project["data_path"],
        output_path=out,
        config_path=trained_project["config_path"],
    )
    assert os.path.exists(out)
    saved = pd.read_csv(out)
    assert "predicted_vehicle_count" in saved.columns
