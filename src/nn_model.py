# ---------------------------------------------------------------
# PRACTICAL 6 — Backpropagation Neural Network
# ---------------------------------------------------------------

"""
A simple feedforward neural network (MLP) for Iris classification.

Commentary:
- XOR logic: A two-layer network with non-linear activations (e.g., sigmoid/ReLU)
  can model XOR by learning non-linear decision boundaries. Extending to Iris
  classification requires multi-class outputs with softmax.

This module builds, trains, and saves a Keras Sequential model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .constants import RANDOM_STATE
from .utils import ensure_dir, project_root, set_seed


# ---------------------------------------------------------------
# PRACTICAL 6 — Build Model
# ---------------------------------------------------------------
def build_mlp(input_dim: int, num_classes: int = 3) -> Any:
    """
    Builds a small MLP for multi-class classification on Iris.

    Architecture
    ----------
    - Dense(16, activation='relu')
    - Dropout(0.1)
    - Dense(8, activation='relu')
    - Dense(num_classes, activation='softmax')
    """
    # Import TensorFlow dynamically to avoid linter warnings when TF is not installed yet.
    import importlib

    tf = importlib.import_module("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_dim=input_dim),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------
# PRACTICAL 6 — Train & Save
# ---------------------------------------------------------------
def train_and_save_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 16,
    model_name: str = "nn_iris.keras",
) -> Tuple[Any, Dict]:
    """
    Trains the MLP with early stopping and saves the model to `saved_models/`.

    Returns
    -------
    (Sequential, Dict)
        Fitted model and a dict of key training parameters.
    """
    set_seed(RANDOM_STATE)
    num_classes = len(np.unique(y_train))

    # Dynamic TF import and utilities
    import importlib

    tf = importlib.import_module("tensorflow")
    to_categorical = tf.keras.utils.to_categorical
    EarlyStopping = tf.keras.callbacks.EarlyStopping

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    model = build_mlp(input_dim=X_train.shape[1], num_classes=num_classes)
    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es],
    )

    save_dir = ensure_dir(project_root() / "saved_models")
    model_path = save_dir / model_name
    model.save(model_path)
    return model, {
        "epochs": epochs,
        "batch_size": batch_size,
        "stopped_epoch": len(history.history["loss"]),
        "model_path": str(model_path),
    }
