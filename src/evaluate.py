# ---------------------------------------------------------------
# EVALUATION — Common for Practicals 2, 4, 5, 6, 7, 8
# ---------------------------------------------------------------

"""
Shared evaluation utilities for regression and classification models,
and a helper to collate model comparison results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    adjusted_rand_score,
)

from .utils import ensure_dir, project_root


# ---------------------------------------------------------------
# EVALUATION — Classification Metrics (Practicals 4, 5, 6, 8)
# ---------------------------------------------------------------
def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Computes standard classification metrics for a multi-class problem (macro-averaged).

    Returns a dictionary with accuracy, precision_macro, recall_macro, and f1_macro.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------
# EVALUATION — Regression Metrics (Practical 2)
# ---------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Computes standard regression metrics: MAE, MSE, and R2.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------
# EVALUATION — Clustering Metrics (Practical 7)
# ---------------------------------------------------------------
def clustering_metrics(X: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """
    Computes clustering diagnostics. If ground-truth labels are provided, computes ARI.
    We always return inertia via the KMeans model (caller passes it) and
    silhouette score can be added if desired.
    """
    metrics = {}
    try:
        from sklearn.metrics import silhouette_score

        metrics["silhouette"] = float(silhouette_score(X, pred_labels))
    except Exception:
        pass
    try:
        metrics["ari"] = float(adjusted_rand_score(labels, pred_labels))
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------
# EVALUATION — Model Comparison Table
# ---------------------------------------------------------------
def models_comparison_table(rows: List[Dict]) -> Path:
    """
    Builds a comparison DataFrame from a list of dict rows and saves it to `outputs/model_comparison.csv`.

    Each dict row should contain at least:
    - 'model' : str
    - relevant metrics like 'f1_macro' (classification) or 'r2' (regression)
    """
    df = pd.DataFrame(rows)
    out_dir = ensure_dir(project_root() / "outputs")
    path = out_dir / "model_comparison.csv"
    df.to_csv(path, index=False)
    return path
