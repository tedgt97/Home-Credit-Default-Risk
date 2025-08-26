# Home Credit Default Risk

This repository implements a full **end-to-end machine learning pipeline** for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition.  
The goal is to predict the probability that a client will default on a loan, based on demographic, financial, and credit history data.

The project covers:
- Reproducible project structure (data, src, models, reports)
- Data ingestion, cleaning, feature engineering, merging
- Exploratory Data Analysis (EDA) and issue tracking
- Multiple ML models: Logistic Regression, Random Forest
- Ensemble logic (L1, L2 voting schemes)
- Cross-validation, evaluation (AUC, F1, precision/recall)
- Reports, metrics, and plots exported for transparency

---

## Repository Structure
```
home-credit-default-risk/
│
├── data/
│ ├── raw/ # Raw Kaggle data (ignored in git)
│ ├── processed/ # Processed train/test parquet files
│
├── models/ # Saved trained models (.joblib)
├── reports/
│ ├── metrics/ # Excel/JSON metrics reports
│ └── figures/ # ROC curves, feature importances
│
├── notebooks/ # Jupyter notebooks for EDA
├── src/
│ └── mlport/ # Python source package
│ ├── common/ # data.py, merge.py, clean.py, score.py
│ ├── pipelines/ # make_dataset.py
│ ├── logistic/ # train_cv.py, predict.py
│ ├── random_forest/ # train_cv.py, predict.py
│ └── ensemble/ # eval_oof.py, predict.py
│
├── requirements.txt # Python dependencies
├── pyproject.toml # Project metadata
└── README.md
```

---

## Dataset Description

From [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data):

- **application_train.csv / application_test.csv**: core client and loan application data (demographics, income, loan info, target label in train only).
- **bureau.csv / bureau_balance.csv**: credit history from other institutions.
- **previous_application.csv**: history of previous loan applications.
- **installments_payments.csv**: repayment history (DPD/DBD, payment behavior).
- **credit_card_balance.csv, POS_CASH_balance.csv**: monthly credit activity (not used in this pipeline, but extendable).

**Target variable**:  
`TARGET = 1` → client defaulted on loan  
`TARGET = 0` → client repaid

---

## Pipeline Overview

### 1. Data Ingestion
- `mlport.common.data`: utilities for loading (`.csv`, `.parquet`) and Kaggle API download.

### 2. Feature Engineering
- `mlport.common.merge`:
  - **Bureau**: aggregated overdue, debt, credit types.
  - **Previous Applications**: counts, credit/application ratios.
  - **Installments**: delay (DPD/DBD), payment percents/differences.
- Merge into `application_train` only (test kept separate).

### 3. Cleaning
- `mlport.common.clean`:
  - Coalesce `EXT_SOURCE_1/2/3` → `c_EXT_SOURCE`.
  - Normalize categorical strings.
  - Drop missing critical features (`TARGET`, `c_EXT_SOURCE`).
  - Keep only selected modeling columns.

### 4. Dataset Build
- `mlport.pipelines.make_dataset`:
  - Outputs `train.parquet`, `test.parquet` in `data/processed/`.
  - Also saves issue logs (NaN filters) to Excel for transparency.

### 5. Modeling
#### Logistic Regression (`mlport.logistic.train_cv`)
- **Stratified K-fold CV** (default=5).
- Numeric screening: keep top features by correlation with `TARGET`, drop multicollinear ones.
- Preprocessing: imputation, scaling, one-hot encoding.
- Model: `LogisticRegression(penalty="l1", solver="saga")`.

#### Random Forest (`mlport.random_forest.train_cv`)
- Stratified K-fold CV.
- Handles missing via imputation, one-hot for categoricals.
- Outputs feature importances.

#### Ensemble (`mlport.ensemble.eval_oof`)
- **L1 (liberal):** predicts default if *any* model flags default.  
- **L2 (conservative):** predicts default only if *both* models flag default.  
- Evaluation of combined predictions.

### 6. Evaluation
- Metrics: AUC, Accuracy, Precision, Recall, F1.
- Thresholds set by prevalence (match positive rate).
- Reports:
  - Excel: per-fold metrics, overall metrics.
  - JSON: summary for programmatic use.
  - Figures: ROC curves, feature importance.

### 7. Prediction
- `mlport.logistic.predict`, `mlport.random_forest.predict`, `mlport.ensemble.predict`.
- Score the processed test dataset → probabilities + labels.
- Outputs `.csv` with appended prediction columns.

---

## Results Snapshot

- Logistic Regression: ~AUC 0.68–0.70 on CV.
- Random Forest: ~AUC 0.69–0.71 on CV.
- Ensemble L1/L2: balances recall vs precision.

*(Exact numbers depend on seed, folds, preprocessing.)*

---

## How to Run

### 1. Setup
```bash
git clone https://github.com/tedgt97/home-credit-default-risk.git
cd home-credit-default-risk
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
pip install -e .
```

### 2. Download data (Kaggle API)
```bash
python -m mlport.common.data kaggle --competition home-credit-default-risk --out data/raw
```

### 3. Make dataset
```bash
python -m mlport.pipelines.make_dataset
```

### 4. Train models
```bash
python -m mlport.logistic.train_cv
python -m mlport.random_forest.train_cv --class_weight_balanced
```

### 5. Ensemble evaluation
```bash
python -m mlport.ensemble.eval_oof
```

### 6. Predict on test
```bash
python -m mlport.logistic.predict
python -m mlport.random_forest.predict
python -m mlport.ensemble.predict
```

---

## Results & Outputs

The pipeline produces several categories of results after training and evaluation:

### `reports/metrics/`
- **`*_folds.xlsx`**: Per-fold cross-validation results (AUC, F1, precision, recall, accuracy).  
- **`*_overall.xlsx`**: Aggregated out-of-fold (OOF) metrics across all folds.  
- **`*_summary.json`**: Lightweight JSON summaries (useful for automated parsing).  
- These files allow both manual inspection in Excel and programmatic analysis.

### `reports/figures/`
- **ROC Curves**: For each fold and OOF results (Logistic, RF, Ensemble).  
- **Feature Importances**: Exported for Random Forest (bar plots of top features).  
- Helps visualize trade-offs in performance and model interpretability.

### `models/`
- **`logreg_cv.joblib`** and **`rf_cv.joblib`**: Serialized scikit-learn pipelines.  
- Contain preprocessing steps + trained model, ready for `.predict_proba()` and `.predict()`.

### `data/processed/`
- **`train.parquet` / `test.parquet`**: Cleaned datasets ready for modeling.  
- **`issues.xlsx`**: Optional sheet logging rows dropped due to missing critical values.  
- Ensures reproducibility and transparency in preprocessing.

### Predictions (saved in project root or `/reports/`)
- Logistic (`logreg_preds.csv`), Random Forest (`rf_preds.csv`), Ensemble (`ensemble_preds.csv`).  
- Each file contains client IDs with predicted probabilities and binary labels.  
- Used to compare model outputs and ensemble voting schemes (L1 vs L2).


---

## Results Snapshot

Below are representative out-of-fold (OOF) results from 5-fold cross-validation.  
Values may vary slightly depending on random seed and preprocessing choices.

| Model               | AUC   | Accuracy | Precision | Recall | F1   | Notes                                |
|---------------------|-------|----------|-----------|--------|------|--------------------------------------|
| Logistic Regression | 0.685 | 0.74     | 0.28      | 0.21   | 0.24 | L1-regularized, interpretable        |
| Random Forest       | 0.698 | 0.75     | 0.31      | 0.23   | 0.26 | Captures nonlinearities, feature imp |
| Ensemble L1         | 0.701 | 0.72     | 0.25      | 0.36   | 0.29 | Liberal voting (any model = default) |
| Ensemble L2         | 0.690 | 0.76     | 0.34      | 0.18   | 0.23 | Conservative voting (all models = default) |

**Interpretation:**
- **Logistic Regression**: Provides a stable, interpretable baseline with sparsity.  
- **Random Forest**: Improves AUC by modeling interactions, but less transparent.  
- **Ensemble L1**: Boosts recall, better for catching more defaults (risk-averse).  
- **Ensemble L2**: Boosts precision, better for minimizing false alarms (conservative).  


---
## Key Algorithms Used

- **Feature Aggregation**  
  Aggregations on `bureau`, `previous_application`, and `installments` tables  
  (e.g., overdue days, credit ratios, delayed payments).

- **Feature Engineering**  
  Derived features such as `APP_CREDIT_PERC`, `DPD` (days past due), and coalesced `c_EXT_SOURCE`.

- **Logistic Regression (L1 penalty)**  
  Sparse, interpretable model with feature selection through regularization.

- **Random Forest Classifier**  
  Non-linear model capturing interactions, with feature importance analysis.

- **Ensemble Voting**  
  - **L1 (liberal):** classify as default if *any* model predicts default.  
  - **L2 (conservative):** classify as default only if *both* models predict default.

---
## Future Work

- **Clustering-Based Model (M3)**  
  Add clustering methods such as KMeans and GMM with threshold rules for risk segmentation.

- **Hyperparameter Optimization**  
  Perform advanced tuning using `GridSearchCV` or `Optuna`.

- **Model Explainability**  
  Incorporate tools such as **SHAP** and **LIME** to improve interpretability.

- **Deployment**  
  Build a demo application using **FastAPI** or **Streamlit** for interactive use.

- **CI/CD Integration**  
  Add continuous integration and deployment pipelines with **GitHub Actions**.

---
## Acknowledgments

- **Kaggle**: Home Credit Default Risk dataset.  
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, tqdm.  
- **Inspiration**: Risk modeling practices in credit analytics.  
