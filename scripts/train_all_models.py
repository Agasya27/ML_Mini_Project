# ---------------------------------------------------------------
# MASTER SCRIPT — Trains and Evaluates All 8 Practicals
# ---------------------------------------------------------------

"""
End-to-end orchestration:
- Load -> Engineer -> Outliers -> Encode -> Split -> Scale
- Train models across all practicals
- Save each model and a comparison CSV under `outputs/`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys
import os
import warnings

import numpy as np
import pandas as pd

# Ensure project root on sys.path for `src` imports when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reduce noisy logs/warnings in batch script
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.constants import RANDOM_STATE, TARGET, TEST_SIZE
from src.data_preprocess import (
    load_iris,
    engineer_features,
    identify_outliers_iqr,
    remove_or_clip_outliers,
    encode_target,
    train_test_split_df,
    scale_features,
)
from src.eda import head_tail_describe, univariate_plots, bivariate_plots, multivariate_plots, correlation_heatmap, pca_plot
from src.models import (
    train_linear_regression,
    train_multiple_linear_regression,
    train_decision_tree,
    train_knn,
    train_kmeans,
    train_random_forest,
    train_svc,
)
from src.nn_model import train_and_save_nn
from src.evaluate import classification_metrics, regression_metrics, clustering_metrics, models_comparison_table
from src.utils import ensure_dir, project_root, save_model, set_seed, plot_confusion_matrix, load_model


def main() -> None:
    # Ensure reproducibility
    set_seed(RANDOM_STATE)

    # Load and engineer features
    df = load_iris()
    df = engineer_features(df)

    # EDA outputs (saved to outputs/)
    head_tail_describe(df)
    univariate_plots(df)
    bivariate_plots(df)
    multivariate_plots(df)
    correlation_heatmap(df)
    pca_plot(df)

    # Handle outliers (clip method by default)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df, outlier_report = remove_or_clip_outliers(df, numeric_cols=numeric_cols, method="clip")

    # Encode target
    df_enc, label_encoder = encode_target(df, target_col=TARGET)

    # Train/test split (classification)
    X_train, X_test, y_train, y_test = train_test_split_df(df_enc, target_col=TARGET)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # PRACTICAL 2 — Regression targets (use continuous targets from dataset)
    # Example 1: Linear Regression predicting 'petal_length' from 'sepal_length'
    df_reg1 = df.copy()
    y_reg1 = df_reg1["petal_length"]
    X_reg1 = df_reg1[["sepal_length"]]
    lr1, lr1_params = train_linear_regression(X_reg1, y_reg1)
    save_model(lr1, "linear_regression_petal_length.joblib")

    # Example 2: Multiple Linear Regression predicting 'petal_length' from all other numeric features
    df_reg2 = df.copy()
    y_reg2 = df_reg2["petal_length"]
    X_reg2 = df_reg2.drop(columns=["petal_length", TARGET])
    mlr, mlr_params = train_multiple_linear_regression(X_reg2, y_reg2)
    save_model(mlr, "multiple_linear_regression_petal_length.joblib")

    # PRACTICAL 4 — Decision Tree
    dt, dt_params = train_decision_tree(X_train_scaled, y_train)
    save_model(dt, "decision_tree.joblib")

    # PRACTICAL 5 — KNN
    knn, knn_params = train_knn(X_train_scaled, y_train)
    save_model(knn, "knn.joblib")

    # PRACTICAL 7 — KMeans (unsupervised, fit on scaled features)
    kmeans, kmeans_params = train_kmeans(pd.concat([X_train_scaled, X_test_scaled], axis=0), k=3)
    save_model(kmeans, "kmeans.joblib")

    # PRACTICAL 8 — Random Forest and SVM
    rf, rf_params = train_random_forest(X_train, y_train)  # tree-based models don't strictly need scaling
    save_model(rf, "random_forest.joblib")

    svc, svc_params = train_svc(X_train_scaled, y_train)
    save_model(svc, "svc.joblib")

    # PRACTICAL 6 — Neural Network (train on scaled data)
    nn_model, nn_info = train_and_save_nn(
        X_train_scaled.values, y_train.values, X_test_scaled.values, y_test.values, model_name="nn_iris.keras"
    )

    # Evaluate and compare models (classification)
    comparison_rows: List[Dict] = []

    # Decision Tree
    y_pred_dt = dt.predict(X_test_scaled)
    metrics_dt = classification_metrics(y_test, y_pred_dt)
    comparison_rows.append({"model": "DecisionTree", **metrics_dt})
    plot_confusion_matrix(y_test, y_pred_dt, labels=sorted(y_train.unique().tolist()), title="Decision Tree")

    # KNN
    y_pred_knn = knn.predict(X_test_scaled)
    metrics_knn = classification_metrics(y_test, y_pred_knn)
    comparison_rows.append({"model": "KNN", **metrics_knn})
    plot_confusion_matrix(y_test, y_pred_knn, labels=sorted(y_train.unique().tolist()), title="KNN")

    # Random Forest
    y_pred_rf = rf.predict(X_test)
    metrics_rf = classification_metrics(y_test, y_pred_rf)
    comparison_rows.append({"model": "RandomForest", **metrics_rf})
    plot_confusion_matrix(y_test, y_pred_rf, labels=sorted(y_train.unique().tolist()), title="Random Forest")

    # SVC
    y_pred_svc = svc.predict(X_test_scaled)
    metrics_svc = classification_metrics(y_test, y_pred_svc)
    comparison_rows.append({"model": "SVC", **metrics_svc})
    plot_confusion_matrix(y_test, y_pred_svc, labels=sorted(y_train.unique().tolist()), title="SVC")

    # KMeans (unsupervised) — Evaluate clustering quality
    kmeans_train_test = pd.concat([X_train_scaled, X_test_scaled], axis=0)
    pred_kmeans_all = kmeans.predict(kmeans_train_test)
    # For ARI, align labels array to pred_kmeans_all length
    y_all = pd.concat([y_train, y_test], axis=0).values
    metrics_kmeans = clustering_metrics(kmeans_train_test.values, y_all, pred_kmeans_all)
    metrics_kmeans.update({"inertia": float(kmeans.inertia_)})
    comparison_rows.append({"model": "KMeans", **metrics_kmeans})

    # Regression metrics for information (not included in F1 comparison)
    y_pred_lr1 = lr1.predict(X_reg1)
    reg1 = regression_metrics(y_reg1, y_pred_lr1)
    comparison_rows.append({"model": "LinearRegression(petal_length~sepal_length)", **reg1})

    y_pred_mlr = mlr.predict(X_reg2)
    reg2 = regression_metrics(y_reg2, y_pred_mlr)
    comparison_rows.append({"model": "MultipleLinearRegression(petal_length~others)", **reg2})

    # Save comparison table
    path_comp = models_comparison_table(comparison_rows)
    print(f"Saved model comparison to: {path_comp}")


if __name__ == "__main__":
    main()
