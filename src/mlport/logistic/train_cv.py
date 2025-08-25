import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold

from mlport.common.data import load_any
from mlport.common.features import split_features

from sklearn.preprocessing import FunctionTransformer

def _replace_inf_with_nan(X):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X

# Helpers
def numeric_corr_screen(X: pd.DataFrame, y: pd.Series, top_k: int = 40, corr_drop: float = 0.95) -> List[str]:
    """
    Rank numeric cols by |corr with y|, then greedily drop any numeric that
    correlates with already-kept ones above corr_drop (to reduce multicollinearity).
    Robust to inf by converting them to NaN before corr.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return []

    Xn = X[num_cols].replace([np.inf, -np.inf], np.nan)

    corrs = Xn.corrwith(y).abs().sort_values(ascending=False)
    ranked = corrs.index.tolist()

    kept = []
    # outer loop: candidate ranking
    for c in tqdm(ranked, desc="Screening numerics", leave=False):
        if len(kept) >= top_k:
            break
        ok = True
        # inner loop: check correlation with already kept
        for k in kept:
            pair = Xn[[c, k]]
            cval = pair.corr().iloc[0, 1]
            if pd.notna(cval) and abs(cval) >= corr_drop:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept



def prevalence_threshold(y_train: pd.Series, val_probs: np.ndarray) -> float:
    """
    Choose threshold so predicted positive rate ~= actual prevalence on the training fold.
    """
    p = (y_train == 1).mean()
    if p <= 0 or p >= 1:
        return 0.5
    return float(np.quantile(val_probs, 1-p))

def metrics_for(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    return {'auc': auc, 'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'threshold': thr}

def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label = "ROC")
    plt.plot([0,1], [0,1], linestyle = "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches = "tight", dpi = 150)
    plt.close()


# Trainer
def main():
    ap = argparse.ArgumentParser(description="Logistic Regression (Stratified CV) with numeric screening + L1 sparsity")
    ap.add_argument("--train", default="data/processed/train.parquet")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # feature-screen settings
    ap.add_argument("--top_k_numeric", type=int, default=40)
    ap.add_argument("--corr_drop", type=float, default=0.95)
    # logistic settings
    ap.add_argument("--C", type=float, default=1.0, help="Inverse reg strength for L1 Logistic (saga)")
    args = ap.parse_args()

    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = reports_dir / "figures"; figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)


    # Load data
    tqdm.write("[LOGREG] Loading data...")
    df = load_any(args.train)
    tqdm.write("[LOGREG] Splitting features...")
    X_all, y_all, num_all, cat_all = split_features(df)

    # quick diagnostics
    num_all = X_all.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = np.isinf(X_all[num_all]).sum()
    bad = inf_counts[inf_counts > 0]
    if len(bad):
        print("[WARN] Columns with +/-inf detected:", bad.to_dict())

    # Numeric screening
    tqdm.write("[LOGREG] Running numeric screening...")
    kept_num = numeric_corr_screen(X_all, y_all, top_k=args.top_k_numeric, corr_drop=args.corr_drop)
    kept_cat = [c for c in cat_all if c in X_all.columns]

    # Preprocessing
    tqdm.write("[LOGREG] Building preprocessors and model...")
    num_pre = Pipeline([
        ("fix_inf", FunctionTransformer(_replace_inf_with_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("varth", VarianceThreshold(0.0)),
    ])
    cat_pre = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ('num', num_pre, kept_num),
            ('cat', cat_pre, kept_cat),
        ],
        remainder="drop"
    )

    # Logistic model (sparse & stable with many one-hots)
    clf = LogisticRegression(
        penalty="l1", solver="saga", C=args.C, max_iter=5000, n_jobs=None
    )

    # CV
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros(len(X_all), dtype=float)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(skf.split(X_all, y_all), total=args.folds, desc="CV folds"), start=1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]

        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Fit with a per‑fold status
        tqdm.write(f"[LOGREG] Fitting fold {fold}...")
        pipe.fit(X_tr, y_tr)

        proba_va = pipe.predict_proba(X_va)[:, 1]
        thr = prevalence_threshold(y_tr, proba_va)
        oof_proba[va_idx] = proba_va

        m = metrics_for(y_va.to_numpy(), proba_va, thr)
        fold_rows.append({"fold": fold, **m})
        tqdm.write(f"[LOGREG][FOLD {fold}] AUC={m['auc']:.4f} F1={m['f1']:.4f} Thr={m['threshold']:.3f}")

        save_roc_plot(y_va.to_numpy(), proba_va, figs_dir / f"logreg_fold{fold}_roc.png", f"LogReg Fold {fold} ROC")
    
    # Save OOF probabilities for ensembling
    oof_df = pd.DataFrame({
        "idx": np.arange(len(X_all)),
        "y_true": y_all.to_numpy(),
        "y_prob_logreg": oof_proba,
    })
    oof_df.to_csv(metrics_dir / "logreg_oof_probs.csv", index=False)

    # Overall OOF metrics (use mean threshold for reporting)
    tqdm.write("[LOGREG] Computing OOF metrics and saving plots...")
    avg_thr = float(np.mean([r["threshold"] for r in fold_rows])) if fold_rows else 0.5
    overall = metrics_for(y_all.to_numpy(), oof_proba, avg_thr)
    print(f"[LOGREG] OOF AUC={overall['auc']:.4f} F1={overall['f1']:.4f} Thr~={avg_thr:.3f}")
    save_roc_plot(y_all.to_numpy(), oof_proba, figs_dir / "logreg_oof_roc.png", "LogReg OOF ROC")

    # Refit on full train & save model
    tqdm.write("[LOGREG] Refit on full data and exporting artifacts...")
    final_pipe = Pipeline([('pre', pre), ('clf', clf)])
    final_pipe.fit(X_all, y_all)
    joblib.dump(final_pipe, models_dir / "logreg_cv.joblib")

    # Export non-zero coefficients for transparency
    # build feature names after preprocessing
    # numeric: kept_num (VarianceThreshold may remove constants but does not rename)
    num_varth = final_pipe.named_steps['pre'].named_transformers_['num'].named_steps['varth']
    support = num_varth.get_support()
    num_out = [c for c, keep in zip(kept_num, support) if keep]
    # categorical: names from OneHotEncoder
    if kept_cat:
        ohe = final_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
        cat_out = ohe.get_feature_names_out(kept_cat).tolist()
    else:
        cat_out = []
    feat_names = np.array(num_out + cat_out)

    coefs = final_pipe.named_steps['clf'].coef_.ravel()
    nz = coefs != 0
    coef_df = pd.DataFrame({'feature': feat_names, 'coef': coefs, 'selected': nz}).sort_values('selected', ascending=False)
    coef_df.to_csv(reports_dir / "logreg_nonzero_coefs.csv", index=False)

    # Save metrics to Excel + JSON
    pd.DataFrame(fold_rows).to_excel(metrics_dir / "logreg_folds.xlsx", index=False)
    pd.DataFrame([overall]).to_excel(metrics_dir / "logreg_overall.xlsx", index=False)
    with open(metrics_dir / "logreg_summary.json", "w") as f:
        json.dump({
            'oof_auc': overall['auc'],
            'oof_f1': overall['f1'],
            'avg_threshold': avg_thr,
            'kept_num': kept_num,
            'kept_cat': kept_cat,
        }, f, indent=2)

    print(f"[LOGREG] Saved model -> {models_dir/'logreg_cv.joblib'}")
    print(f"[LOGREG] Metrics & plots -> {reports_dir}/(metrics|figures)")

if __name__ == "__main__":
    main()
