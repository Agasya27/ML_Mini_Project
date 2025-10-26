# ---------------------------------------------------------------
# IrisSuite — 8 Practicals on the Iris Dataset (VS Code-Ready)
# ---------------------------------------------------------------

IrisSuite is a production-quality, well-documented Python project that implements eight classic Machine Learning practicals using the Iris dataset — plus data preprocessing, EDA, hyperparameter tuning, model comparison, model saving, and a Streamlit GUI.

This repository is designed to be easy to run locally and to explain in a viva. Every file and major section begins with a clear comment header stating which practical(s) it implements.

## Practicals Implemented

- PRACTICAL 1 — Data Preprocessing & EDA
  - Files:
    - `src/constants.py` — PRACTICAL 1 — Constants and Configuration
    - `src/data_preprocess.py` — PRACTICAL 1 — Data Preprocessing (Handling Missing, Duplicates, Outliers, Encoding, Scaling)
    - `src/eda.py` — PRACTICAL 1 — Exploratory Data Analysis
- PRACTICAL 2 — Linear Regression & Multiple Linear Regression
  - File: `src/models.py` (train_linear_regression, train_multiple_linear_regression)
- PRACTICAL 3 — Find-S Algorithm
  - File: `src/find_s.py`
- PRACTICAL 4 — Decision Tree Classifier
  - File: `src/models.py` (train_decision_tree)
- PRACTICAL 5 — K-Nearest Neighbors (KNN)
  - File: `src/models.py` (train_knn)
- PRACTICAL 6 — Backpropagation Neural Network (Feedforward)
  - File: `src/nn_model.py`
- PRACTICAL 7 — K-Means Clustering
  - File: `src/models.py` (train_kmeans)
- PRACTICAL 8 — Random Forest and Support Vector Machine (SVM)
  - File: `src/models.py` (train_random_forest, train_svc)

Additional shared functionality:
- `src/evaluate.py` — EVALUATION — Common for Practicals 2, 4, 5, 6, 7, 8
- `src/utils.py` — UTILITIES — Used Across All Practicals
- `scripts/train_all_models.py` — MASTER SCRIPT — Trains and Evaluates All 8 Practicals
- `scripts/evaluate_models.py` — EVALUATION SCRIPT — Compare All Models
- `streamlit_app/app.py` — GUI — Streamlit Interface for Model Interaction

## Innovation Section
Implemented in `src/data_preprocess.py` as part of `engineer_features()`:
- `sepal_area = sepal_length * sepal_width`
- `petal_area = petal_length * petal_width`
- `petal_area_ratio = petal_area / max(sepal_area, 1e-8)` — safe divide-by-zero handling
- `large_sepal_flag = int(sepal_length > mean_sepal_length and sepal_width > mean_sepal_width)`
- `petal_to_sepal_ratio = (petal_length + petal_width) / (sepal_length + sepal_width)` — captures overall petal-to-sepal proportion, which often correlates with species.
- `symmetry_index = |sepal_length - petal_length| / (sepal_length + petal_length)` — expresses balance/symmetry in flower shape; different species show different symmetry.

Why these help: the ratio aggregates petal vs. sepal magnitude into a stable signal, and the symmetry index highlights shape balance; together, they can improve class separability beyond raw lengths/widths.

The engineered dataset (including these features) is saved to `data/iris_pp.csv`.

## Streamlit UI
The app provides a modern, clean, and interactive interface:
- Gradient theme (light/dark toggle) and custom CSS for a polished look.
- Tabs:
  - Overview & About — project summary, student details, and the Innovation Section explanation.
  - EDA & Model Insights — pre-generated EDA images from `outputs/`, model comparison table and chart, per-model metrics (accuracy/precision/recall/F1), confusion matrix heatmap, and feature importance (DT/RF).
  - Prediction — sliders (or CSV upload), automatic engineered features, predictions with probability bars, and a download button for batch predictions.

Screenshot: Insert Streamlit app screenshot here

## Evaluation Criteria Mapping
- Classification models (Practicals 4, 5, 6, 8): accuracy, precision (macro), recall (macro), F1 (macro), confusion matrix.
- Regression models (Practical 2): MAE, MSE, R².
- Clustering (Practical 7): inertia and silhouette score (if labels available, ARI is also computed).
- Model comparison table saved to `outputs/model_comparison.csv`.

### Coursework Mapping
- Problem Analysis ✅
- Data Preprocessing & EDA ✅
- Model Development & Hyperparameter Tuning ✅
- Innovation (new features + enhanced UI) ✅
- Knowledge & Demonstration (explained in About tab) ✅

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open the folder `IrisSuite` in VS Code.

## How to Run

- Train and evaluate all models (saves models and comparison table):
  ```
  python scripts/train_all_models.py
  ```
- Evaluate saved models and regenerate comparison plots:
  ```
  python scripts/evaluate_models.py
  ```
- Launch the Streamlit GUI:
  ```
  streamlit run streamlit_app/app.py
  ```

## Notes

- All models and preprocessing assets (scaler, label encoder) are saved in `saved_models/`.
- Plots are saved in `outputs/`.
- Randomness is controlled via `RANDOM_STATE = 42` from `src/constants.py`.
- Train/test split uses `stratify=y` for reproducibility.

## Repository Structure

IrisSuite/
├─ data/
│  └─ iris.csv
├─ src/
│  ├─ constants.py
│  ├─ data_preprocess.py
│  ├─ eda.py
│  ├─ find_s.py
│  ├─ models.py
│  ├─ nn_model.py
│  ├─ evaluate.py
│  ├─ utils.py
│  └─ __init__.py
├─ scripts/
│  ├─ train_all_models.py
│  └─ evaluate_models.py
├─ streamlit_app/
│  └─ app.py
├─ saved_models/
├─ outputs/
├─ requirements.txt
├─ README.md
└─ .gitignore
