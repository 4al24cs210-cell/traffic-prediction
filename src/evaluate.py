"""
evaluate.py
-----------
Compute traffic-flow model performance metrics and save report artefacts.

Metrics
-------
    MAE   — Mean Absolute Error
    RMSE  — Root Mean Squared Error
    MAPE  — Mean Absolute Percentage Error (epsilon-guarded for zero targets)

Public API
----------
    compute_metrics(y_true, y_pred)             -> dict
    save_metrics(metrics, path)                 -> None
    plot_predictions(y_true, y_pred, timestamps, path) -> None
    plot_residuals(y_true, y_pred, path)        -> None
    run_evaluation(y_true, y_pred, timestamps, cfg) -> dict
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return a dict with MAE, RMSE, and MAPE.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted vehicle counts.

    Returns
    -------
    dict with keys: mae, rmse, mape
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE: guard against division by zero.
    # Use a threshold of 0.01 (1% of normalised scale) to avoid inflated MAPE
    # when y_true values are near zero after Min-Max normalisation.
    threshold = np.maximum(np.abs(y_true), 0.01)
    mape = float(np.mean(np.abs((y_true - y_pred) / threshold)) * 100)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, path: str) -> None:
    """Serialise *metrics* to a JSON file at *path*.

    Parameters
    ----------
    metrics : dict
    path : str
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[evaluate] Metrics saved -> {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[Sequence] = None,
    path: str = "reports/prediction_vs_actual.png",
    max_points: int = 500,
) -> None:
    """Save a line chart comparing actual vs. predicted traffic flow.

    Parameters
    ----------
    y_true, y_pred : array-like
    timestamps : sequence, optional
        X-axis labels. If None, uses integer indices.
    path : str
        Output PNG file path.
    max_points : int
        Subsample if series is longer (keeps plot readable).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Subsample for readability
    if len(y_true) > max_points:
        idx = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        x_vals = idx if timestamps is None else np.asarray(timestamps)[idx]
    else:
        x_vals = np.arange(len(y_true)) if timestamps is None else np.asarray(timestamps)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x_vals, y_true, label="Actual", linewidth=1.2, color="#2563EB")
    ax.plot(x_vals, y_pred, label="Predicted", linewidth=1.2, color="#DC2626", linestyle="--")
    ax.set_title("Actual vs. Predicted Traffic Flow")
    ax.set_xlabel("Time")
    ax.set_ylabel("Vehicle Count")
    ax.legend()
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluate] Prediction plot saved -> {path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str = "reports/residuals.png",
) -> None:
    """Save a residual distribution histogram.

    Parameters
    ----------
    y_true, y_pred : array-like
    path : str
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, color="#7C3AED", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Residuals Distribution (Actual − Predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluate] Residuals plot saved -> {path}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[Sequence],
    cfg: dict,
) -> dict:
    """Run the full evaluation pipeline: metrics + plots + JSON save.

    Parameters
    ----------
    y_true, y_pred : array-like
    timestamps : sequence or None
    cfg : dict
        Parsed config.yaml.

    Returns
    -------
    dict
        Metric dict (mae, rmse, mape).
    """
    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 40)
    print("  EVALUATION RESULTS")
    print("=" * 40)
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAPE : {metrics['mape']:.4f} %")
    print("=" * 40 + "\n")

    reports = cfg.get("reports", {})
    save_metrics(metrics, reports.get("metrics_path", "reports/metrics.json"))
    plot_predictions(
        y_true, y_pred, timestamps,
        path=reports.get("predictions_plot", "reports/prediction_vs_actual.png"),
    )
    plot_residuals(
        y_true, y_pred,
        path=reports.get("residuals_plot", "reports/residuals.png"),
    )

    return metrics
