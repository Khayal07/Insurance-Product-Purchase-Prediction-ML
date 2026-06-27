# Multi-Output Insurance Purchase Prediction Pipeline

A production-ready, end-to-end Machine Learning pipeline designed to predict customer choices across 7 distinct insurance policy components (A through G) simultaneously. Built using sequential customer shopping transaction logs, this project leverages advanced feature engineering and multi-output ensemble frameworks to achieve a phenomenal **92.53% Overall Average Accuracy**.

---

## 🚀 Project Overview

In the insurance industry, understanding and predicting customer behavior during the quote-generation funnel is critical for dynamic pricing and targeted cross-selling. This project utilizes the transaction logs of the famous *Allstate Purchase Prediction Challenge* on Kaggle. 

Instead of treating the 7 policy options as isolated problems, this repository implements a **Multi-Output Classification** framework that captures the underlying correlations between different sub-policies, optimizing the joint prediction vector.

### 📊 Key Performance Metrics
* **Champion Model:** XGBoost Classifier (Overall Average: **92.53%**)
* **Runner-Up Model:** CatBoost Classifier (Overall Average: **92.51%**)
* **Baseline Model:** Random Forest Classifier (Overall Average: **92.49%**)
* **Component Peak:** Target **D** achieved the highest single accuracy at **95.04%**.
* **Minimum Individual Baseline:** Target **G** represents the lowest ceiling but still maintains a strong **86.85% - 87.01%** accuracy floor.

---

## 🛠️ Data Pipeline Architecture

The project is modularly structured into three sequential Jupyter Notebooks to ensure clean code division, readability, and reproducibility:

### 1. `01_EDA.ipynb` (Exploratory Data Analysis)
* Statistical profiling of transaction lengths, demographic variables, and price states.
* Correlation analysis between historical quote components and the final purchased policy.
* Visualizing class distributions for all 7 multi-class target variables.

### 2. `02_Preprocessing.ipynb` (The Feature Engineering Engine)
* **The 92% Accuracy Catalyst:** Extracted and isolated the `last_shopping_point` (the final quote a user reviewed right before checking out). 
* Engineered rolling historical aggregate features to track price volatility and policy change frequencies per customer journey.
* Handled missing structural values (e.g., `risk_factor`) using robust median/mode mapping.
* Optimized memory consumption and encoded categorical columns uniformly.

### 3. `03_Modeling.ipynb` (Multi-Output Framework)
* Transformed target labels to a uniform global 0-index matrix to ensure strict compatibility with gradient boosters.
* Implemented parallelized `MultiOutputClassifier` structures.
* Evaluated and bench-marked three industry-standard algorithms:
  * **Random Forest Classifier** (Baseline)
  * **XGBoost Classifier** (Advanced Gradient Boosting - **Project Winner**)
  * **CatBoost Classifier** (Categorical Boosting)
* Plotted comprehensive visual performance comparisons using Seaborn.

---

## 📈 Model Performance Benchmark (Real Results)

The following table summarizes the exact test set accuracy scores achieved across all 7 insurance options:

| Target Component | Random Forest (Accuracy) | XGBoost (Accuracy) 🏆 | CatBoost (Accuracy) |
| :--- | :---: | :---: | :---: |
| **A** | 0.9272 | 0.9280 | 0.9276 |
| **B** | 0.9353 | 0.9353 | 0.9353 |
| **C** | 0.9313 | 0.9319 | 0.9317 |
| **D** | 0.9503 | 0.9503 | 0.9504 |
| **E** | 0.9362 | 0.9358 | 0.9362 |
| **F** | 0.9253 | 0.9258 | 0.9261 |
| **G** | 0.8690 | 0.8701 | 0.8685 |
| 📊 **OVERALL AVERAGE** | **0.9249** | **0.9253** | **0.9251** |

> **Business Insight:** All three models perform exceptionally close to each other, maintaining a strict ~92.5% average. This proves that the feature engineering workflow—specifically capturing the `last_shopping_point`—retains the definitive signal of customer choice. While XGBoost takes the micro-lead at **92.53%**, any of the tuned boosting architectures are highly suitable for deployment.

---

## 💻 Tech Stack & Frameworks

* **Language:** Python 3.10+
* **Core ML Libraries:** `scikit-learn`, `xgboost`, `catboost`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

---

## ⚙️ How To Run This Project

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/insurance-purchase-prediction.git](https://github.com/yourusername/insurance-purchase-prediction.git)
   cd insurance-purchase-prediction

2. **Install Dependencies:**
   ```bash
    pip install -r requirements.txt


3. **Execute the Pipeline:**
    -Run the notebooks in chronological order to generate data assets and train models:

    -Execute 01_EDA.ipynb to analyze the distribution.

    -Run 02_Preprocessing.ipynb to build preprocessed_features.csv.

    -Run 03_Modeling.ipynb to train models, visualize performance, and view results.


4. **Future Improvements**
    -Hyperparameter Tuning: Implement Optuna to optimize XGBoost's max_depth and learning_rate to push beyond the 92.5% threshold.

    -API Deployment: Wrap the winning Multi-Output framework with FastAPI to serve real-time multi-label vector predictions.