# ---------------------------------------------------------------
# PRACTICAL 1 — Exploratory Data Analysis
# ---------------------------------------------------------------

"""
EDA utilities that produce and save plots into `outputs/`.
Each function returns a list of saved file paths for traceability.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

from .utils import ensure_dir, project_root
from .constants import TARGET


def _outputs_dir() -> Path:
    """Returns the outputs directory, ensuring it exists."""
    return ensure_dir(project_root() / "outputs")


# ---------------------------------------------------------------
# PRACTICAL 1 — Head/Tail/Describe
# ---------------------------------------------------------------
def head_tail_describe(df: pd.DataFrame, n: int = 5) -> List[Path]:
    """
    Saves head, tail, and describe as CSV snapshots for quick inspection.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.
    n : int
        Number of rows for head and tail.

    Returns
    -------
    List[Path]
        Paths to the saved files.
    """
    out_dir = _outputs_dir()
    head_path = out_dir / "head.csv"
    tail_path = out_dir / "tail.csv"
    desc_path = out_dir / "describe.csv"

    df.head(n).to_csv(head_path, index=False)
    df.tail(n).to_csv(tail_path, index=False)
    df.describe(include="all").to_csv(desc_path)

    return [head_path, tail_path, desc_path]


# ---------------------------------------------------------------
# PRACTICAL 1 — Univariate Plots
# ---------------------------------------------------------------
def univariate_plots(df: pd.DataFrame) -> List[Path]:
    """
    Plots histograms and KDEs for numeric features and saves them.
    """
    out_dir = _outputs_dir()
    paths: List[Path] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="teal")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        fig.tight_layout()
        path = out_dir / f"univariate_{col}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------
# PRACTICAL 1 — Bivariate Plots
# ---------------------------------------------------------------
def bivariate_plots(df: pd.DataFrame) -> List[Path]:
    """
    Plots pairwise scatter plots with hue as species (if present).
    """
    out_dir = _outputs_dir()
    paths: List[Path] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if TARGET in df.columns:
        hue = TARGET
    else:
        hue = None

    # Pairplot can be heavy; save a single composite
    g = sns.pairplot(df[numeric_cols + ([hue] if hue else [])], hue=hue, corner=True, diag_kind="hist")
    path = out_dir / "pairplot.png"
    g.fig.suptitle("Pairplot", y=1.02)
    g.savefig(path, dpi=150)
    plt.close(g.fig)
    paths.append(path)
    return paths


# ---------------------------------------------------------------
# PRACTICAL 1 — Multivariate Plots
# ---------------------------------------------------------------
def multivariate_plots(df: pd.DataFrame) -> List[Path]:
    """
    Creates a boxplot per feature grouped by species (if available).
    """
    out_dir = _outputs_dir()
    paths: List[Path] = []
    if TARGET in df.columns:
        for col in df.select_dtypes(include="number").columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
            ax.set_title(f"{col} by {TARGET}")
            fig.tight_layout()
            path = out_dir / f"box_{col}_by_{TARGET}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


# ---------------------------------------------------------------
# PRACTICAL 1 — Correlation Heatmap
# ---------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame) -> List[Path]:
    """
    Computes and plots the correlation matrix heatmap for numeric features.
    """
    out_dir = _outputs_dir()
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------
# PRACTICAL 1 — PCA Plot
# ---------------------------------------------------------------
def pca_plot(df: pd.DataFrame, n_components: int = 2) -> List[Path]:
    """
    Performs PCA on numeric features and plots the first two components,
    colored by species if available.
    """
    out_dir = _outputs_dir()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    X = df[numeric_cols].values
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    if TARGET in df.columns:
        sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=df[TARGET], palette="tab10", ax=ax)
    else:
        sns.scatterplot(x=comps[:, 0], y=comps[:, 1], ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D)")
    fig.tight_layout()
    path = out_dir / "pca_2d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]
