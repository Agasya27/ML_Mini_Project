# ---------------------------------------------------------------
# EVALUATION SCRIPT — Compare All Models
# ---------------------------------------------------------------

"""
Loads saved models, evaluates them on the test split (recomputed in the same way),
and writes `outputs/model_comparison.csv` and an F1 comparison bar chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on sys.path for `src` imports when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reduce noisy logs/warnings in batch script
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.constants import TARGET, RANDOM_STATE
from src.data_preprocess import load_iris, engineer_features, remove_or_clip_outliers, encode_target, train_test_split_df, scale_features
from src.evaluate import classification_metrics, models_comparison_table
from src.utils import project_root, load_model, ensure_dir, plot_confusion_matrix, set_seed


def main() -> None:
    set_seed(RANDOM_STATE)

    # Rebuild the same preprocessing pipeline to create the same split
    df = load_iris()
    df = engineer_features(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df, _ = remove_or_clip_outliers(df, numeric_cols=numeric_cols, method="clip")
    df_enc, _ = encode_target(df, target_col=TARGET)
    X_train, X_test, y_train, y_test = train_test_split_df(df_enc, target_col=TARGET)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # Load classifiers
    dt = load_model("decision_tree.joblib")
    knn = load_model("knn.joblib")
    rf = load_model("random_forest.joblib")
    svc = load_model("svc.joblib")
    # XGBoost may not be present if not trained
    xgb_path = project_root() / "saved_models" / "xgboost.joblib"
    xgb = load_model("xgboost.joblib") if xgb_path.exists() else None

    rows: List[Dict] = []

    models_to_eval = [
        ("DecisionTree", dt, X_test_scaled),
        ("KNN", knn, X_test_scaled),
        ("RandomForest", rf, X_test),  # RF uses unscaled
        ("SVC", svc, X_test_scaled),
    ]
    if xgb is not None:
        models_to_eval.append(("XGBoost", xgb, X_test))

    for name, model, X_eval in models_to_eval:
        y_pred = model.predict(X_eval)
        metrics = classification_metrics(y_test, y_pred)
        rows.append({"model": name, **metrics})
        plot_confusion_matrix(y_test, y_pred, labels=sorted(y_train.unique().tolist()), title=name)

    comp_path = models_comparison_table(rows)
    print(f"Saved comparison: {comp_path}")

    # F1 comparison bar chart
    df_comp = pd.DataFrame(rows)
    out_dir = ensure_dir(project_root() / "outputs")
    fig, ax = plt.subplots(figsize=(6, 4))
    # Use a single color to avoid seaborn FutureWarning about palette without hue
    sns.barplot(data=df_comp, x="model", y="f1_macro", ax=ax, color=sns.color_palette("viridis", n_colors=1)[0])
    ax.set_title("F1 Macro — Model Comparison")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    fig.tight_layout()
    chart_path = out_dir / "f1_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved F1 chart: {chart_path}")


if __name__ == "__main__":
    main()
