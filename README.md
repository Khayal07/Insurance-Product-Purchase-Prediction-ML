# Multi-Output Insurance Purchase Prediction Pipeline

An end-to-end, leakage-free Machine Learning pipeline that predicts customer choices across 7 distinct insurance policy components (A through G) simultaneously. Built on the sequential customer shopping transaction logs of the *Allstate Purchase Prediction Challenge* (Kaggle), this project combines behavioral feature engineering with multi-output gradient boosting — and, crucially, evaluates every model against the honest baseline that dominates this dataset.

---

## 🚀 Project Overview

In the insurance industry, understanding and predicting customer behavior during the quote-generation funnel is critical for dynamic pricing and targeted cross-selling. Each customer in the data views a sequence of quotes (shopping points) before buying; the final purchase is recorded separately.

Instead of treating the 7 policy options as isolated problems, this repository implements a **Multi-Output Classification** framework, and instead of reporting accuracy in a vacuum, it measures every model against the **"last quote" baseline** — simply predicting that the customer buys exactly what they saw in their last viewed quote. That baseline already reaches ~92.5% per-option accuracy, so the real scientific question of this project is: *can a model learn when a customer will deviate from their last quote?*

### 📊 Key Performance Metrics (held-out test set, 19,402 customers)
* **Naive Baseline (Last Quote):** 92.48% average per-option accuracy — the honest reference point.
* **Champion Model:** Tuned XGBoost, **92.54%** average per-option accuracy, beating the baseline on 6 of 7 components.
* **Hardest Component:** Target **G** — baseline 86.75% → model **87.04%**. This is where customers change their minds most often, and where the model adds the most value.
* **Strict Metric (All-7 Exact Match):** ~68.9% for both the baseline and the best models — an honest reminder of how dominant the last-quote signal is (even winning Kaggle solutions barely moved this metric).
* **Stability:** 3-fold cross-validation on G: 0.866 ± 0.002; train-test gap ≤ 1pp on every target (no overfitting).

---

## 🛠️ Data Pipeline Architecture

The project is modularly structured into three sequential Jupyter Notebooks:

### 1. `01_EDA.ipynb` (Exploratory Data Analysis)
* Statistical profiling of transaction lengths, demographic variables, and price states.
* Correlation heatmap of the numeric features.
* Class distributions for all 7 multi-class target variables.
* **The baseline discovery:** how often the purchased option equals the last viewed quote (~92.5% per option, ~68.8% for the whole package) — the finding that shapes the entire evaluation strategy.

### 2. `02_Preprocessing.ipynb` (The Feature Engineering Engine)
* Extracts the `last_shopping_point` (the final quote a user reviewed before checking out) as the current-state feature vector.
* Engineers historical aggregate features per customer journey: cost statistics (mean/min/max/std), price volatility, and cost change from first to last quote.
* **Option dynamics features:** for each option A-G, how many distinct values the customer viewed (`_nunique`) and how many times the option switched between consecutive quotes (`_n_changes`) — the signal that tells the model *when to distrust the last quote*.
* Deterministic missing-value fills only (`C_previous`/`duration_previous` → 0, `car_value` → 'Unknown'). Fitted transformations (median imputation, categorical encoding) are deliberately deferred to the modeling notebook so they can be learned from the train split only — **no data leakage**.

### 3. `03_Modeling.ipynb` (Multi-Output Framework & Honest Evaluation)
* Train/test split **first**, then train-only fitting of `risk_factor` median imputation and `state`/`car_value` encodings.
* Trains and benchmarks Random Forest, XGBoost, and CatBoost inside `MultiOutputClassifier` wrappers, side by side with the naive baseline.
* **Hyperparameter tuning** with `RandomizedSearchCV` on the hardest target (G), 3-fold cross-validation, and a per-target overfitting check.
* **Deep dive on target G:** classification report (macro F1), normalized confusion matrix, baseline-vs-model comparison split by customer history stability, feature importance, and **SHAP interpretability**.
* **Inference demo:** predicting the full 7-option package for an unseen customer and decoding labels back to the original scale.

---

## 📈 Model Performance Benchmark (Real Results)

Test set accuracy across all 7 insurance options, including the baseline every model must beat:

| Target Component | Baseline: Last Quote | Random Forest | XGBoost 🏆 | CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 0.9272 | 0.9274 | 0.9276 | 0.9275 |
| **B** | 0.9353 | 0.9353 | 0.9353 | 0.9352 |
| **C** | 0.9316 | 0.9312 | 0.9320 | 0.9317 |
| **D** | 0.9502 | 0.9502 | 0.9504 | 0.9503 |
| **E** | 0.9362 | 0.9362 | 0.9363 | 0.9362 |
| **F** | 0.9260 | 0.9245 | 0.9260 | 0.9260 |
| **G** | 0.8675 | 0.8651 | **0.8701** | 0.8682 |
| 📊 **OVERALL AVERAGE** | 0.9248 | 0.9243 | **0.9254** | 0.9250 |
| 🎯 **ALL-7 EXACT MATCH** | 0.6889 | 0.6842 | 0.6884 | 0.6888 |

The tuned XGBoost (n_estimators=150, max_depth=6, learning_rate=0.1) keeps the 0.9254 overall average and raises G to **0.8704**.

> **Business Insight:** The margins over the baseline are small in absolute terms — and that is the honest, well-known reality of this dataset. The model's real value is concentrated where it matters: on target G and on customers with *unstable* shopping histories (baseline 83.8% → model 84.2%), exactly the customers a recommendation system needs help with. For everyone else, "show them their last quote" is already near-optimal, and the model correctly learns to reproduce it.

---

## 💻 Tech Stack & Frameworks

* **Language:** Python 3.10+
* **Core ML Libraries:** `scikit-learn`, `xgboost`, `catboost`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Interpretability:** `shap`

---

## ⚙️ How To Run This Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Khayal07/Insurance-Product-Purchase-Prediction-ML.git
   cd Insurance-Product-Purchase-Prediction-ML
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the Pipeline (order matters):**
    - Run `01_EDA.ipynb` to analyze the distributions and the baseline.
    - Run `02_Preprocessing.ipynb` to build `preprocessed_features.csv` / `preprocessed_targets.csv`.
    - Run `03_Modeling.ipynb` to train models, tune, evaluate, and view all visualizations.
    - Note: `03` must always be run after `02` — the fitted preprocessing steps (imputation, encoding) live in `03` by design.

4. **Future Improvements**
    - Sequence models (e.g., customer-journey transformers) to squeeze more signal out of the quote-order dynamics.
    - Joint prediction of the 7 options (e.g., classifier chains) to directly optimize the all-7 exact match metric.
    - API Deployment: wrap the winning Multi-Output framework with FastAPI to serve real-time multi-label vector predictions (the inference demo in `03` is the blueprint).
