# ---------------------------------------------------------------
# STREAMLIT UI — Implements Interactive GUI for All Practicals
# ---------------------------------------------------------------

"""
Interactive GUI to load trained models, input features (via sliders or CSV upload),
compute engineered features, run predictions, and display comparison/evaluation artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import re

# Ensure project root is on sys.path so `src` imports work when launching via Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reduce noisy third-party warnings/logging in UI
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Hide TF INFO/WARN
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.constants import TARGET, RANDOM_STATE, TEST_SIZE
from src.data_preprocess import engineer_features, load_iris, encode_target, train_test_split_df, scale_features
from src.utils import project_root, load_model
from src.evaluate import classification_metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin


# ---------------------------------------------------------------
# GUI — Helpers
# ---------------------------------------------------------------
def list_saved_models() -> Dict[str, Path]:
    """
    Lists available models in saved_models/ and returns a mapping name -> path.
    Filters out non-estimator assets like label encoders and scalers.
    """
    sm_dir = project_root() / "saved_models"
    models: Dict[str, Path] = {}
    for p in sorted(sm_dir.glob("*")):
        # Only include recognized model file types
        if p.suffix.lower() in {".joblib", ".keras"}:
            # Exclude preprocessing assets
            if p.name.lower() in {"label_encoder.joblib", "scaler.joblib"}:
                continue
            models[p.name] = p
    return models


def list_classifiers() -> Dict[str, Path]:
    """Return only classifier models supported in the UI (exclude regressors and clustering)."""
    all_models = list_saved_models()
    allowed = {"decision_tree.joblib", "knn.joblib", "random_forest.joblib", "svc.joblib"}
    cls_models: Dict[str, Path] = {}
    for name, path in all_models.items():
        if name in allowed or name.endswith(".keras"):
            cls_models[name] = path
    return cls_models


def _keras_load_model(path: Path):
    """Loads a Keras model via dynamic TensorFlow import to avoid static import warnings."""
    import importlib

    tf = importlib.import_module("tensorflow")
    return tf.keras.models.load_model(path)


def prepare_data_for_inference(df_source: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame | None]:
    """Prepare assets for evaluation/prediction using the uploaded dataset if provided.

    If df_source is None, falls back to the built-in Iris dataset. If the uploaded
    dataset does not include the target column, metrics/CM will be disabled (y_* = None).
    """
    if df_source is None:
        df = load_iris()
    else:
        df = df_source.copy()

    df = engineer_features(df)

    if TARGET in df.columns:
        # Local encoding (do NOT save encoder to disk)
        df_enc = df.copy()
        le = LabelEncoder()
        df_enc[TARGET] = le.fit_transform(df_enc[TARGET])
        X_train, X_test, y_train, y_test = train_test_split_df(df_enc, target_col=TARGET)
    else:
        # No target: split features only (metrics will be disabled)
        X = df.copy()
        X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        y_train = None
        y_test = None

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_s,
        "X_test_scaled": X_test_s,
        "scaler": scaler,
    }


def build_input_sidebar() -> pd.DataFrame:
    """
    Sidebar sliders for raw Iris feature input. Engineered features are computed downstream.
    """
    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.8, 0.1, key="sl")
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.0, 0.1, key="sw")
    petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.35, 0.1, key="pl")
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 1.3, 0.1, key="pw")
    df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    return df


def _normalize_name(col: str) -> str:
    """Normalize a column name to compare against known Iris feature names.
    Lowercase, remove non-alphanumerics.
    """
    return re.sub(r"[^a-z0-9]+", "", col.strip().lower())


def coerce_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV and coerce columns to required Iris feature names.

    Returns the four canonical feature columns and, if present, a target column mapped to 'species'.

    Expected canonical feature columns:
    - sepal_length, sepal_width, petal_length, petal_width

    Target detection (optional):
    - Accepts common synonyms like species, class, target, variety (any case/spacing)

    Strategy:
    1) Rename using a normalization map and common synonyms.
    2) If still missing features, fall back to first 4 numeric columns and rename in required order with a warning.
    """
    df = pd.read_csv(uploaded_file)

    # Map of normalized variants to canonical feature names
    synonyms = {
        "sepallength": "sepal_length",
        "sepallengthcm": "sepal_length",
        "sepallengthcms": "sepal_length",
        "sepallengthmm": "sepal_length",
        "sepallengthin": "sepal_length",
        "sepallengthsepal_length": "sepal_length",
        "sepallengthsepalwidth": "sepal_length",
        "sepallengthsepal": "sepal_length",
        "sepallengthsepalwidthpetallengthpetalwidth": "sepal_length",
        "sepallengthsepalwidthpetallength": "sepal_length",
        "sepallengthsepalwidth": "sepal_length",
        "sepallengthcmsepalwidthcm": "sepal_length",
        "sepal_length": "sepal_length",

        "sepalwidth": "sepal_width",
        "sepalwidthcm": "sepal_width",
        "sepal_width": "sepal_width",

        "petallength": "petal_length",
        "petallengthcm": "petal_length",
        "petal_length": "petal_length",

        "petalwidth": "petal_width",
        "petalwidthcm": "petal_width",
        "petal_width": "petal_width",

        # sklearn original names
        "sepal length cm": "sepal_length",
        "sepal width cm": "sepal_width",
        "petal length cm": "petal_length",
        "petal width cm": "petal_width",
        "sepallengthcmsepalwidthcm": "sepal_length",
    }

    # Potential target label synonyms (normalized)
    target_synonyms = {"species", "class", "target", "variety", "label", "speciesname", "species_type"}

    normalized = {_normalize_name(c): c for c in df.columns}
    rename_map: Dict[str, str] = {}
    detected_target_original: str | None = None

    # Detect feature columns and build rename map
    for norm, original in normalized.items():
        if norm in synonyms:
            rename_map[original] = synonyms[norm]
    # Detect target column and plan rename to 'species'
    for norm, original in normalized.items():
        if norm in target_synonyms:
            detected_target_original = original
            rename_map[original] = TARGET  # map to 'species'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Fallback: try sklearn-style columns exactly
        sklearn_map = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
        overlap = [c for c in sklearn_map if c in df.columns]
        if overlap:
            df = df.rename(columns={c: sklearn_map[c] for c in overlap})
            missing = [c for c in required if c not in df.columns]

    if missing:
        # Final fallback: select first 4 numeric columns and assume the canonical order
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 4:
            picked = num_cols[:4]
            st.warning(
                "Uploaded CSV headers not recognized; using the first 4 numeric columns as sepal_length, sepal_width, petal_length, petal_width."
            )
            coerced = df[picked].copy()
            coerced.columns = required
            # If a target column was detected elsewhere, append it
            if TARGET in df.columns:
                coerced[TARGET] = df[TARGET].values
            elif detected_target_original and detected_target_original in df.columns:
                coerced[TARGET] = df[detected_target_original].values
            return coerced
        else:
            raise KeyError(
                "Uploaded CSV must contain sepal_length, sepal_width, petal_length, petal_width or at least four numeric columns."
            )

    # Build final view: features + optional species
    cols = required.copy()
    if TARGET in df.columns:
        cols.append(TARGET)
    return df[cols].copy()


def compute_engineered(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Computes engineered features to match training-time feature set.
    """
    df_eng = engineer_features(df_raw)
    # Drop target if present in accidental uploads
    if TARGET in df_eng.columns:
        df_eng = df_eng.drop(columns=[TARGET])
    return df_eng


def _normalize_species_label(val: str) -> str:
    """Normalize species labels to compare against the classic Iris classes.
    - Lowercase, remove non-letters, and strip leading 'iris' if present
    """
    s = re.sub(r"[^a-z]", "", str(val).strip().lower())
    if s.startswith("iris"):
        s = s[len("iris"):]
    return s


def validate_iris_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the uploaded dataframe looks like the Iris dataset.

    Checks:
    - Required feature columns present: sepal_length, sepal_width, petal_length, petal_width
    - Feature columns are numeric
    - At least 90% of rows fall within plausible Iris ranges:
      sepal_length [4.0,8.0], sepal_width [2.0,4.5], petal_length [1.0,7.0], petal_width [0.1,2.5]
    - If a target column is present, class names must be a subset of {setosa, versicolor, virginica}
      (accepts variants like 'Iris-setosa')
    """
    errors: List[str] = []

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return False, errors

    # Type check
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Columns must be numeric: {', '.join(non_numeric)}")

    # Range coverage check
    plausible_ranges = {
        "sepal_length": (4.0, 8.0),
        "sepal_width": (2.0, 4.5),
        "petal_length": (1.0, 7.0),
        "petal_width": (0.1, 2.5),
    }
    out_of_range_counts = {}
    total = len(df)
    for c, (lo, hi) in plausible_ranges.items():
        within = df[c].between(lo, hi, inclusive="both").sum()
        coverage = within / max(total, 1)
        if coverage < 0.9:  # allow some outliers but expect majority within known ranges
            out_of_range_counts[c] = total - within
    if out_of_range_counts:
        details = ", ".join(f"{k}:{v} out-of-range" for k, v in out_of_range_counts.items())
        errors.append(f"Too many values outside typical Iris ranges — {details}")

    # Target class check (if present)
    if TARGET in df.columns:
        allowed = {"setosa", "versicolor", "virginica"}
        vals = set(_normalize_species_label(v) for v in df[TARGET].dropna().unique())
        if not vals.issubset(allowed):
            unknown = sorted(list(vals.difference(allowed)))
            errors.append(f"Unknown species labels found: {', '.join(unknown)}")

    return (len(errors) == 0), errors


def model_predict(model_name: str, X_row: pd.DataFrame, assets: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Loads the selected model and produces predictions and class probabilities if supported.
    """
    model_path = list_saved_models()[model_name]
    if model_name.endswith(".keras"):
        # Neural network (Keras)
        model = _keras_load_model(model_path)
        # Align and scale using training scaler
        train_cols = list(assets["X_train"].columns)
        aligned = X_row.reindex(columns=train_cols)
        scaler = assets.get("scaler", None)
        if scaler is None:
            raise RuntimeError("Scaler not available in assets; cannot scale input for neural network.")
        X_scaled = scaler.transform(aligned.values)
        probs = model.predict(X_scaled, verbose=0)[0]
        pred_int = int(np.argmax(probs))
        return {"pred": pred_int, "probs": probs.tolist()}
    else:
        # Sklearn models
        model = load_model(model_name)
        # Align columns to model's training order if available
        feature_order = list(getattr(model, "feature_names_in_", assets["X_train"].columns))
        aligned = X_row.reindex(columns=feature_order)

        # Choose scaling based on training behavior
        if "random_forest" in model_name:
            X_pred_df = aligned
        else:
            scaler = assets.get("scaler", None)
            if scaler is None:
                # Fall back to already-scaled structure if scaler missing
                X_pred_df = assets["X_train_scaled"].iloc[:1, :].copy()
                X_pred_df.loc[X_pred_df.index[0]] = aligned.values[0]
                X_pred_df = X_pred_df.reindex(columns=feature_order)
            else:
                X_scaled = scaler.transform(aligned.values)
                X_pred_df = pd.DataFrame(X_scaled, columns=feature_order)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pred_df.values)[0]
            pred_int = int(np.argmax(probs))
            return {"pred": pred_int, "probs": probs.tolist()}
        else:
            pred_int = int(model.predict(X_pred_df.values)[0])
            return {"pred": pred_int, "probs": []}


def main() -> None:
    # Page config and theming
    st.set_page_config(page_title="IrisSuite — Streamlit GUI", layout="wide")

    # Sidebar — Theme and Menu
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Navigate", ["Home", "EDA", "Predict", "About"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.write("Upload CSV (optional) with columns: sepal_length,sepal_width,petal_length,petal_width")
    uploaded = st.sidebar.file_uploader("CSV Upload", type=["csv"])
    st.sidebar.markdown("---")
    # Manual input sliders always available
    df_sidebar = build_input_sidebar()
    if st.sidebar.button("Reset Inputs"):
        for k in ["sl", "sw", "pl", "pw"]:
            if k in st.session_state:
                del st.session_state[k]

    # Inject premium CSS (dark theme only)
    bg_css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f1020 0%, #103036 100%);
            color: #e6f0f2;
        }
        .main > div { padding-top: 0rem; }
        h1, h2, h3 { color: #d4e3ff; }
        .metric-container { background: rgba(0,0,0,0.35); border-radius: 12px; padding: 1rem; }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)

    # Hero header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="font-size: 2rem;">🌸</div>
      <div>
        <h1 style="margin-bottom:0">IrisSuite — Intelligent Flower Classifier</h1>
        <div style="margin-top:-8px; opacity:0.8"> • EDA • Tuning • Streamlit GUI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prepare raw input from uploaded CSV if provided; else allow manual predictions using built-in Iris
    df_raw = None
    if uploaded is not None:
        df_raw = coerce_uploaded_dataframe(uploaded)
        # Block non-Iris datasets explicitly
        is_valid, errs = validate_iris_dataset(df_raw)
        if not is_valid:
            st.error("Dataset not matching with the current environment. Please upload the classic Iris dataset.")
            with st.expander("Why was it blocked?", expanded=False):
                for e in errs:
                    st.write("- " + e)
            st.stop()

    # Prepare assets (train/test and scalers) based on uploaded data (or built-in if none)
    assets = prepare_data_for_inference(df_raw)
    models_all = list_saved_models()
    models = list_classifiers()

    # Navigation-controlled content
    if menu == "Home":
        st.markdown("## 🌺 Overview & About")
        st.write("""
        This app demonstrates eight ML practicals on the classic Iris dataset with a modern, interactive UI.
        It includes end-to-end preprocessing, EDA, regression/classification/clustering models, hyperparameter tuning,
        model comparison, and a neural network — all wrapped in a clean Streamlit interface.
        """)

        st.markdown("Developed by Agasya Butolia, Roll No. 66, Shri Ramdeobaba College of Engineering and Management")

        st.markdown("### 🧪 Innovation Section — Extra Engineered Features")
        st.write("""
        - petal_to_sepal_ratio = (petal_length + petal_width) / (sepal_length + sepal_width) — captures overall petal-to-sepal proportion, often species-specific.
        - symmetry_index = |sepal_length − petal_length| / (sepal_length + petal_length) — expresses shape balance/symmetry of the flower.
        """)
        st.info("These features aim to improve separability between species, complementing the classic measurements.")

    elif menu == "EDA":
        st.markdown("## 📊 EDA & Model Insights")
        out_dir = project_root() / "outputs"
        cols = st.columns(3)
        eda_files = [
            (out_dir / "correlation_heatmap.png", "Correlation Heatmap"),
            (out_dir / "pairplot.png", "Pairplot"),
            (out_dir / "pca_2d.png", "PCA (2D)"),
        ]
        for i, (p, caption) in enumerate(eda_files):
            if p.exists():
                with cols[i % 3]:
                    st.image(str(p), caption=caption, use_container_width=True)
        comp_path = out_dir / "model_comparison.csv"
        if comp_path.exists():
            df_comp = pd.read_csv(comp_path)
            st.markdown("### 🤖 Compare Models")
            st.dataframe(df_comp, use_container_width=True)
            # Interactive bar chart of accuracy
            if "accuracy" in df_comp.columns:
                st.bar_chart(df_comp.set_index("model")["accuracy"], use_container_width=True)
        else:
            st.info("Run training/evaluation to generate model comparison table in outputs/.")

        # Model selection for insights
        st.markdown("### 🔎 Model Insights")
        if models:
            model_name = st.selectbox("Choose a model", options=list(models.keys()), key="insights_model")
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                from sklearn.metrics import confusion_matrix

                # Evaluate selected model on test set for metrics cards
                if assets["y_test"] is None:
                    st.info("Uploaded CSV has no labels ('species'); metrics and confusion matrix are unavailable.")
                else:
                    if model_name.endswith(".keras"):
                        model = _keras_load_model(models[model_name])
                        cols = assets["X_train_scaled"].columns
                        X_s = assets["X_test_scaled"][cols]
                        y_true = assets["y_test"].values
                        y_pred = np.argmax(model.predict(X_s.values, verbose=0), axis=1)
                    else:
                        model = load_model(model_name)
                        feature_names = getattr(model, "feature_names_in_", None)
                        if "random_forest" in model_name:
                            X_df = assets["X_test"].copy()
                        else:
                            X_df = assets["X_test_scaled"].copy()
                        if feature_names is not None:
                            X_df = X_df.reindex(columns=list(feature_names))
                        y_true = assets["y_test"].values
                        y_pred = model.predict(X_df.values)

                    mets = classification_metrics(y_true, y_pred)
                    with st.container():
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{mets['accuracy']:.2f}")
                        m2.metric("Precision (macro)", f"{mets['precision_macro']:.2f}")
                        m3.metric("Recall (macro)", f"{mets['recall_macro']:.2f}")
                        m4.metric("F1 (macro)", f"{mets['f1_macro']:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # PRACTICAL 4 — Decision Tree Visualized Here
                    # PRACTICAL 8 — Random Forest Visualized Here
                    with st.expander("Confusion Matrix", expanded=True):
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        st.pyplot(fig)

                with st.expander("Model Details / Hyperparameters", expanded=False):
                    if 'model' in locals():
                        if hasattr(model, "get_params"):
                            st.json(model.get_params())
                        else:
                            st.write("(Keras model)")
                    else:
                        st.write("(Model details unavailable because the uploaded CSV has no labels.)")

                # Feature importance (Decision Tree / Random Forest)
                if 'model' in locals() and hasattr(model, "feature_importances_"):
                    fi = pd.Series(model.feature_importances_, index=assets["X_train"].columns)
                    with st.expander("Feature Importance", expanded=True):
                        st.bar_chart(fi.sort_values(ascending=False))
            except Exception as e:
                st.warning(f"Could not render insights for '{model_name}'. Details: {e}")
        else:
            st.info("No saved models found. Train models to enable insights.")

    elif menu == "Predict":
        st.markdown("## 🤖 Prediction")
        if models:
            model_name_pred = st.selectbox("Choose a model", options=list(models.keys()), key="predict_model")
        else:
            st.warning("No saved models found — train models first.")
            model_name_pred = None
        # Input source selection
        opts = ["Manual (sidebar sliders)"] + (["Uploaded CSV"] if df_raw is not None else [])
        source = st.radio("Input source", options=opts, horizontal=True)

        if source == "Manual (sidebar sliders)":
            df_infer_manual = compute_engineered(df_sidebar)
            cols_view = st.columns(2)
            with cols_view[0]:
                st.markdown("#### Manual Input (from Sidebar)")
                st.dataframe(df_sidebar, use_container_width=True)
            with cols_view[1]:
                st.markdown("#### Engineered Features Preview")
                st.dataframe(df_infer_manual, use_container_width=True)
            if model_name_pred and st.button("Predict (Manual Input)", type="primary"):
                try:
                    result = model_predict(model_name_pred, df_infer_manual, assets)
                    pred_int = result["pred"]
                    probs = result["probs"]
                    from src.utils import load_model as _load
                    le = _load("label_encoder.joblib")
                    pred_label = le.inverse_transform([pred_int])[0]
                    st.success(f"Predicted species: {pred_label}")
                    if probs:
                        classes = le.inverse_transform(np.arange(len(probs)))
                        prob_df = pd.DataFrame({"class": classes, "probability": probs})
                        st.bar_chart(prob_df.set_index("class"))
                except Exception as e:
                    st.error(f"Failed to predict with selected model '{model_name_pred}'. Please choose another model.\nDetails: {e}")
        else:
            # Uploaded CSV path
            df_infer_csv = compute_engineered(df_raw)
            cols_view = st.columns(2)
            with cols_view[0]:
                st.markdown("#### Input (Uploaded CSV)")
                st.dataframe(df_raw, use_container_width=True)
            with cols_view[1]:
                st.markdown("#### Engineered Features Preview")
                st.dataframe(df_infer_csv, use_container_width=True)

            if model_name_pred and st.button("Predict (First Row of CSV)", type="primary"):
                try:
                    result = model_predict(model_name_pred, df_infer_csv.iloc[[0]], assets)
                    pred_int = result["pred"]
                    probs = result["probs"]
                    from src.utils import load_model as _load
                    le = _load("label_encoder.joblib")
                    pred_label = le.inverse_transform([pred_int])[0]
                    st.success(f"Predicted species: {pred_label}")
                    if probs:
                        classes = le.inverse_transform(np.arange(len(probs)))
                        prob_df = pd.DataFrame({"class": classes, "probability": probs})
                        st.bar_chart(prob_df.set_index("class"))
                except Exception as e:
                    st.error(f"Failed to predict with selected model '{model_name_pred}'. Please choose another model.\nDetails: {e}")

            # Batch predictions and download
            if df_raw is not None and model_name_pred:
                try:
                    from src.utils import load_model as _load
                    le = _load("label_encoder.joblib")
                    preds = []
                    df_infer_all = compute_engineered(df_raw)
                    for i in range(df_infer_all.shape[0]):
                        res = model_predict(model_name_pred, df_infer_all.iloc[[i]], assets)
                        preds.append(res["pred"])
                    labels = le.inverse_transform(np.array(preds))
                    out_df = df_raw.copy()
                    out_df["predicted_species"] = labels
                    csv_bytes = out_df.to_csv(index=False).encode()
                    st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions.csv")
                except Exception as e:
                    st.warning(f"Could not generate batch predictions. Details: {e}")

    elif menu == "About":
        st.markdown("## ℹ️ About")
        st.write("IrisSuite — an educational yet production-quality mini-project showcasing 8 ML practicals on the Iris dataset with a modern Streamlit UI.")
        st.markdown("Developed by Agasya Butolia, Roll No. 66, Shri Ramdeobaba College of Engineering and Management")

    # Footer
    st.markdown("---")
    st.markdown(
        "_Developed by Agasya Butolia, Roll No. 66, Shri Ramdeobaba College of Engineering and Management_"
    )


if __name__ == "__main__":
    main()
