# Model Card — Credit Default Risk (M1/M2 + L1/L2)

**Data**: Home Credit (public competition).  
**Target**: DEFAULT (binary).  
**Features**: application + engineered aggregates (bureau, previous apps, installments).  
**Train/Val**: Stratified K-Fold (k=5), OOF metrics reported.

## Models
- **M1**: Logistic Regression (L1, saga), screening top K numerics, OHE for categoricals.
- **M2**: Random Forest (n_estimators=400, class_weight balanced).
- **Ensembles**:
  - **L1**: OR (at least one default).
  - **L2**: AND (both default).

## Metrics (OOF)
See `reports/metrics/*_overall.xlsx` and `ensemble_summary.xlsx`.

## Thresholding
- Prevalence-matched per fold, averaged → `avg_threshold` used for test inference.

## Interpretability
- M1: `logreg_nonzero_coefs.csv`
- M2: `rf_feature_importances.csv`

## Limitations
- Public dataset; no temporal validation.
- No cost-sensitive optimization yet; calibration optional.

## Repro
See README “Quickstart”.
