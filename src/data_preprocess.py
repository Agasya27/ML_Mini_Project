# ---------------------------------------------------------------
# PRACTICAL 1 — Data Preprocessing (Handling Missing, Duplicates, Outliers, Encoding, Scaling)
# ---------------------------------------------------------------

"""
Data preprocessing utilities:
- Load the Iris dataset from `data/iris.csv` (fall back to sklearn if needed)
- Feature engineering
- IQR outlier detection (bounds, identification, handling)
- Target encoding
- Train/Test split with stratification
- Feature scaling

Saves the engineered dataset to `data/iris_pp.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE, TEST_SIZE, TARGET
from .utils import ensure_dir, project_root, save_model


# ---------------------------------------------------------------
# PRACTICAL 1 — Loading
# ---------------------------------------------------------------
def load_iris() -> pd.DataFrame:
    """
    Loads the Iris dataset from `data/iris.csv`. If the file is not found,
    falls back to sklearn's built-in dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sepal_length, sepal_width, petal_length, petal_width, species
    """
    data_path = project_root() / "data" / "iris.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        from sklearn.datasets import load_iris as sk_load_iris

        data = sk_load_iris(as_frame=True)
        df = data.frame.rename(
            columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
            }
        )
        df["species"] = df["target"].map(dict(enumerate(data.target_names)))
        df = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
    # Remove duplicates and handle any missing values conservatively
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna()
    return df


# ---------------------------------------------------------------
# PRACTICAL 1 — Feature Engineering
# ---------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Adds engineered features:
        - sepal_area = sepal_length * sepal_width
        - petal_area = petal_length * petal_width
        - petal_area_ratio = petal_area / max(sepal_area, 1e-8) (safe divide-by-zero)
        - large_sepal_flag = 1 if sepal_length > mean(sepal_length) and sepal_width > mean(sepal_width) else 0
        - petal_to_sepal_ratio = (petal_length + petal_width) / (sepal_length + sepal_width)
            (Intuition: captures overall petal-to-sepal proportion; helpful for species separation.)
        - symmetry_index = abs(sepal_length - petal_length) / (sepal_length + petal_length)
            (Intuition: measures shape balance/symmetry of flower; different species exhibit different symmetry.)

    Saves engineered DataFrame to `data/iris_pp.csv`.

    Returns
    -------
    pd.DataFrame
        The engineered DataFrame.
    """
    df = df.copy()
    # Compute areas
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_area"] = df["petal_length"] * df["petal_width"]

    # Safe ratio (avoid divide by zero by clamping denominator)
    denom = np.maximum(df["sepal_area"].values, 1e-8)
    df["petal_area_ratio"] = df["petal_area"].values / denom

    # Binary flag for "large sepal"
    sepal_len_mean = df["sepal_length"].mean()
    sepal_wid_mean = df["sepal_width"].mean()
    df["large_sepal_flag"] = ((df["sepal_length"] > sepal_len_mean) & (df["sepal_width"] > sepal_wid_mean)).astype(int)

    # Additional innovative features
    # petal_to_sepal_ratio: overall petal-to-sepal proportion (correlates with species)
    denom_len_wid = np.maximum((df["sepal_length"] + df["sepal_width"]).values, 1e-8)
    df["petal_to_sepal_ratio"] = (df["petal_length"].values + df["petal_width"].values) / denom_len_wid

    # symmetry_index: balance between sepal and petal lengths (shape symmetry indicator)
    denom_sym = np.maximum((df["sepal_length"] + df["petal_length"]).values, 1e-8)
    df["symmetry_index"] = np.abs(df["sepal_length"].values - df["petal_length"].values) / denom_sym

    # Save engineered dataset (includes new features)
    save_path = project_root() / "data" / "iris_pp.csv"
    ensure_dir(save_path.parent)
    df.to_csv(save_path, index=False)
    return df


# ---------------------------------------------------------------
# PRACTICAL 1 — IQR Outlier Handling
# ---------------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """
    Computes IQR-based lower and upper bounds for outlier detection.

    Steps:
    - Compute Q1 (25th percentile) and Q3 (75th percentile)
    - IQR = Q3 - Q1
    - Lower bound = Q1 - k * IQR
    - Upper bound = Q3 + k * IQR

    Parameters
    ----------
    series : pd.Series
        Numeric series to compute bounds for.
    k : float, default=1.5
        Multiplier for the IQR range.

    Returns
    -------
    (float, float)
        Lower and upper bounds.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def identify_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, pd.Series]:
    """
    Identifies outliers for each numeric column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scan.
    numeric_cols : List[str]
        Numeric columns to evaluate.

    Returns
    -------
    Dict[str, pd.Series]
        Mapping of column -> boolean series marking outliers.
    """
    outlier_masks: Dict[str, pd.Series] = {}
    for col in numeric_cols:
        lower, upper = iqr_bounds(df[col])
        outlier_masks[col] = (df[col] < lower) | (df[col] > upper)
    return outlier_masks


def remove_or_clip_outliers(
    df: pd.DataFrame, numeric_cols: List[str], method: str = "clip"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Handles outliers detected by IQR either by clipping to bounds or removing rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : List[str]
        Numeric columns to consider.
    method : str, default='clip'
        'clip' to clamp values to [lower, upper], 'remove' to drop outlier rows.

    Returns
    -------
    (pd.DataFrame, Dict[str, Dict[str, float]])
        Cleaned DataFrame and a per-column report with bounds.
    """
    df = df.copy()
    report: Dict[str, Dict[str, float]] = {}
    if method not in {"clip", "remove"}:
        raise ValueError("method must be either 'clip' or 'remove'")

    if method == "clip":
        for col in numeric_cols:
            lower, upper = iqr_bounds(df[col])
            report[col] = {"lower": float(lower), "upper": float(upper)}
            # Clamp values to bounds to reduce outlier influence
            df[col] = df[col].clip(lower=lower, upper=upper)
    else:
        # Remove any row that is an outlier in ANY numeric column
        mask = pd.Series(False, index=df.index)
        for col in numeric_cols:
            lower, upper = iqr_bounds(df[col])
            report[col] = {"lower": float(lower), "upper": float(upper)}
            mask |= (df[col] < lower) | (df[col] > upper)
        df = df[~mask].reset_index(drop=True)

    return df, report


# ---------------------------------------------------------------
# PRACTICAL 1 — Encoding, Split, Scaling
# ---------------------------------------------------------------
def encode_target(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encodes the target label column with a LabelEncoder and saves it.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    target_col : str
        Name of the target column to encode.

    Returns
    -------
    (pd.DataFrame, LabelEncoder)
        DataFrame with encoded target column and the fitted encoder.
    """
    df = df.copy()
    if target_col not in df:
        raise KeyError(f"Target column '{target_col}' not in DataFrame.")

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    # Persist the encoder for inference in the GUI
    save_model(le, "label_encoder.joblib")
    return df, le


def train_test_split_df(
    df: pd.DataFrame, target_col: str = TARGET, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into train/test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    target_col : str
        Name of the target column for stratification.
    test_size : float
        Test set fraction.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature and target splits as DataFrame/Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales features using StandardScaler and saves the scaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, StandardScaler)
        Scaled X_train, X_test and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    # Persist the scaler
    save_model(scaler, "scaler.joblib")
    return X_train_scaled, X_test_scaled, scaler
