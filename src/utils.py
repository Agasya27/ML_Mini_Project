# ---------------------------------------------------------------
# UTILITIES — Used Across All Practicals
# ---------------------------------------------------------------

"""
Utility helpers used across the repository:
- save_model, load_model: persist sklearn models with joblib
- plot_confusion_matrix: plot confusion matrix with seaborn/matplotlib
- set_seed: set seeds for numpy, random, and tensorflow
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import RANDOM_STATE


def project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path | str) -> Path:
    """
    Ensures the directory exists; creates it if not, and returns Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model: Any, filename: str) -> Path:
    """
    Saves a model object using joblib to the `saved_models/` directory.

    Parameters
    ----------
    model : Any
        The model object (e.g., sklearn estimator) to persist.
    filename : str
        The filename (e.g., 'decision_tree.joblib').

    Returns
    -------
    Path
        The full path to the saved file.
    """
    sm_dir = ensure_dir(project_root() / "saved_models")
    path = sm_dir / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str) -> Any:
    """
    Loads a persisted model from `saved_models/`.

    Parameters
    ----------
    filename : str
        The filename to load.

    Returns
    -------
    Any
        The loaded object.
    """
    path = project_root() / "saved_models" / filename
    return joblib.load(path)


def plot_confusion_matrix(y_true, y_pred, labels, title: str, save_path: Optional[Path] = None) -> Path:
    """
    Plots and saves a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (integers or strings).
    y_pred : array-like
        Predicted labels.
    labels : list
        List of class labels in the display order.
    title : str
        Title of the plot.
    save_path : Optional[Path]
        If provided, path to save the figure; otherwise saved to outputs/.

    Returns
    -------
    Path
        The path where the figure is saved.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is None:
        save_dir = ensure_dir(project_root() / "outputs")
        save_path = save_dir / f"{title.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Sets random seeds for reproducibility across numpy, random, and tensorflow.

    Notes
    -----
    TensorFlow is imported dynamically via importlib to avoid static import
    warnings if it's not installed yet. If unavailable, we silently skip.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import importlib

        tf = importlib.import_module("tensorflow")
        # Some TF variants may not expose random.set_seed; guard just in case.
        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed)
    except Exception:
        # TensorFlow may be unavailable in some environments; fail quietly.
        pass
