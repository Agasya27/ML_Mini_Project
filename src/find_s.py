# ---------------------------------------------------------------
# PRACTICAL 3 — Find-S Algorithm
# ---------------------------------------------------------------

"""
Implementation of the Find-S algorithm on the Iris dataset.

We discretize continuous features into 3 bins: 'low', 'medium', 'high'.
The algorithm learns the maximally specific hypothesis from positive examples.

For a multi-class dataset like Iris, choose a target class (default 'setosa').
"""

from __future__ import annotations

from typing import List, Tuple, Sequence
import pandas as pd
import numpy as np

from .constants import TARGET


# ---------------------------------------------------------------
# PRACTICAL 3 — Discretization
# ---------------------------------------------------------------
def discretize_features(df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """
    Discretizes numeric columns into equal-frequency bins with labels low/medium/high.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric columns.
    bins : int
        Number of bins.

    Returns
    -------
    pd.DataFrame
        DataFrame with discretized features (categorical).
    """
    df_disc = df.copy()
    numeric_cols = df_disc.select_dtypes(include="number").columns.tolist()
    labels = ["low", "medium", "high"][:bins]
    for col in numeric_cols:
        # Use qcut for equal-frequency bins; handle duplicates by fallback to cut.
        try:
            df_disc[col] = pd.qcut(df_disc[col], q=bins, labels=labels, duplicates="drop")
        except ValueError:
            df_disc[col] = pd.cut(df_disc[col], bins=bins, labels=labels, include_lowest=True)
    return df_disc


# ---------------------------------------------------------------
# PRACTICAL 3 — Find-S Core
# ---------------------------------------------------------------
def find_s(df_disc: pd.DataFrame, positive_class: str = "setosa") -> List[str]:
    """
    Runs Find-S to derive the maximally specific hypothesis from positive examples.

    The hypothesis is a list of attribute constraints; each element is either a specific value
    or '?' meaning "any". We initialize with the first positive example and generalize only when needed.

    Parameters
    ----------
    df_disc : pd.DataFrame
        Discretized DataFrame containing categorical features and the target column.
    positive_class : str
        The class considered as 'positive'.

    Returns
    -------
    List[str]
        The learned hypothesis as a list of feature values or '?'.
    """
    # Use only features (exclude target)
    X_cols = [c for c in df_disc.columns if c != TARGET]
    positives = df_disc[df_disc[TARGET] == positive_class]
    if positives.empty:
        raise ValueError(f"No examples with positive class '{positive_class}' found.")
    # Start with the first positive example
    hypothesis = list(positives.iloc[0][X_cols].astype(str).values)

    # For each positive example, generalize hypothesis where needed
    for _, row in positives.iterrows():
        for i, col in enumerate(X_cols):
            if hypothesis[i] != str(row[col]) and hypothesis[i] != "?":
                hypothesis[i] = "?"
    return hypothesis


def predict_from_hypothesis(hypothesis: Sequence[str], x: Sequence[str]) -> bool:
    """
    Applies the learned hypothesis to a single discretized instance.

    Parameters
    ----------
    hypothesis : Sequence[str]
        The hypothesis list from `find_s`.
    x : Sequence[str]
        A discretized instance (categorical values as strings).

    Returns
    -------
    bool
        True if the instance matches the hypothesis (i.e., predicted positive).
    """
    for h, xv in zip(hypothesis, x):
        if h != "?" and h != xv:
            return False
    return True
