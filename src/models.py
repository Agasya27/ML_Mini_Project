# ---------------------------------------------------------------
# PRACTICALS 2, 4, 5, 7, 8 — Model Implementations (Regression, Decision Tree, KNN, KMeans, RF, SVM)
# ---------------------------------------------------------------

"""
Wrapper training functions for models used across multiple practicals.
Classification models leverage GridSearchCV with cv=5 and scoring='f1_macro'.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .constants import RANDOM_STATE


# ---------------------------------------------------------------
# PRACTICAL 2 — Linear Regression
# ---------------------------------------------------------------
def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[LinearRegression, Dict]:
    """
    PRACTICAL 2 — Trains a simple Linear Regression to predict a single continuous target
    (e.g., predicting petal_length from sepal_length).

    Notes
    -----
    LinearRegression has no hyperparameters to tune via GridSearchCV; we return empty best_params.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model, {}


def train_multiple_linear_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[LinearRegression, Dict]:
    """
    PRACTICAL 2 — Trains a Multiple Linear Regression using all features to predict a continuous target.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model, {}


# ---------------------------------------------------------------
# PRACTICAL 4 — Decision Tree Classifier
# ---------------------------------------------------------------
def train_decision_tree(X: pd.DataFrame, y: pd.Series) -> Tuple[DecisionTreeClassifier, Dict]:
    """
    PRACTICAL 4 — Trains a Decision Tree Classifier with GridSearchCV.

    Returns
    -------
    (DecisionTreeClassifier, Dict)
        Fitted best estimator and best_params.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 2, 3, 4, 5, 6, 8],
        "min_samples_split": [2, 4, 6],
    }
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------
# PRACTICAL 5 — K-Nearest Neighbors
# ---------------------------------------------------------------
def train_knn(X: pd.DataFrame, y: pd.Series) -> Tuple[KNeighborsClassifier, Dict]:
    """
    PRACTICAL 5 — Trains a KNN Classifier with GridSearchCV.
    """
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    base = KNeighborsClassifier()
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------
# PRACTICAL 7 — K-Means Clustering
# ---------------------------------------------------------------
def train_kmeans(X: pd.DataFrame, k: int = 3) -> Tuple[KMeans, Dict]:
    """
    PRACTICAL 7 — Trains K-Means clustering (unsupervised).
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    model.fit(X)
    return model, {"n_clusters": k, "n_init": 10}


# ---------------------------------------------------------------
# PRACTICAL 8 — Random Forest and SVM
# ---------------------------------------------------------------
def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, Dict]:
    """
    PRACTICAL 8 — Trains a Random Forest Classifier with GridSearchCV.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 2, 3, 4, 5, 6, 8],
        "min_samples_split": [2, 4, 6],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def train_svc(X: pd.DataFrame, y: pd.Series) -> Tuple[SVC, Dict]:
    """
    PRACTICAL 8 — Trains an SVM (SVC) classifier with GridSearchCV.

    Notes
    -----
    probability=True is set for downstream probability estimates in the GUI.
    """
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear", "poly"],
    }
    base = SVC(random_state=RANDOM_STATE, probability=True)
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_
