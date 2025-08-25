import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold

from mlport.common.data import load_any
from mlport.common.features import split_features


# Helpers
def _replace_inf_with_nan(X):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X

def numeric_corr_screen(X: pd.DataFrame, y: pd.Series, top_k: int=40, corr_drop: float=0.95) -> List[str]:
    """
    Rank numeric cols by |corr with y|, then greedily drop any numeric that correlates with already-kept ones above corr_drop (to reduce multicollinearity).
    Robust to inf by converting them to NaN before corr.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return []
    
    Xn = X[num_cols].replace([np.inf, -np.inf], np.nan)

    corrs = Xn.corrwith(y).abs().sort_values(ascending=False)
    ranked = corrs.index.tolist()

    kept = []
    for c in tqdm(ranked, desc="Screening numerics", leave=False):
        if len(kept) >= top_k:
            break
        ok = True
        for k in kept:
            pair = Xn[[c,k]]
            cval = pair.corr().iloc[0,1]
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
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"auc": auc, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": thr}

def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# Trainer
def main():
    ap = argparse.ArgumentParser(description="Random Forest (Stratified CV) with numeric screening + tqdm")
    ap.add_argument("--train", default="data/processed/train.parquet")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # feature-screen settings
    ap.add_argument("--top_k_numeric", type=int, default=40)
    ap.add_argument("--corr_drop", type=float, default=0.95)
    # RF settings
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--class_weight_balanced", action="store_true", help="Use class_weight='balanced' to counter class imbalance")
    args = ap.parse_args()

    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = reports_dir / "figures"; figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write("[RF] Loading data...")
    df = load_any(args.train)

    tqdm.write("[RF] Splitting features...")
    X_all, y_all, num_all, cat_all = split_features(df)

    #diagnostic for inf
    num_all = X_all.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = np.isinf(X_all[num_all]).sum()
    bad = inf_counts[inf_counts > 0]
    if len(bad):
        tqdm.write(f"[WARN][RF] Columns with +/-inf detected: {bad.to_dict()}")
    
    tqdm.write("[RF] Running numeric screening...")
    kept_num = numeric_corr_screen(X_all, y_all, top_k=args.top_k_numeric, corr_drop=args.corr_drop)
    kept_cat = [c for c in cat_all if c in X_all.columns]

    tqdm.write("[RF] Building preprocessors and model...")
    num_pre = Pipeline([
        ("fix_inf", FunctionTransformer(_replace_inf_with_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        # RF does not need scaling, but harmless; remove StandardScaler()
        ("varth", VarianceThreshold(0.0)),
    ])
    cat_pre = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pre, kept_num),
            ("cat", cat_pre, kept_cat),
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.seed,
        class_weight=("balanced" if args.class_weight_balanced else None),
    )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros(len(X_all), dtype=float)
    fold_rows = []

    tqdm.write("[RF] Starting CV...")
    for fold, (tr_idx, va_idx) in enumerate(tqdm(skf.split(X_all, y_all), total=args.folds, desc="CV folds"), start=1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]

        pipe = Pipeline([("pre", pre), ("clf", rf)])

        tqdm.write(f"[RF] Fitting fold {fold}...")
        pipe.fit(X_tr, y_tr)

        proba_va = pipe.predict_proba(X_va)[:,1]
        thr = prevalence_threshold(y_tr, proba_va)
        oof_proba[va_idx] = proba_va

        m = metrics_for(y_va.to_numpy(), proba_va, thr)
        fold_rows.append({"fold": fold, **m})
        tqdm.write(f"[RF][FOLD {fold}] AUC={m['auc']:.4f} F1={m['f1']:.4f} Thr={m['threshold']:.3f}")

        save_roc_plot(y_va.to_numpy(), proba_va, figs_dir / f"rf_fold{fold}_roc.png", f"RF Fold {fold} ROC")

     # Save OOF probabilities for ensembling
    oof_df = pd.DataFrame({
        "idx": np.arange(len(X_all)),
        "y_true": y_all.to_numpy(),
        "y_prob_rf": oof_proba,
    })
    oof_df.to_csv(metrics_dir / "rf_oof_probs.csv", index=False)


    avg_thr = float(np.mean([r["threshold"] for r in fold_rows])) if fold_rows else 0.5
    overall = metrics_for(y_all.to_numpy(), oof_proba, avg_thr)
    tqdm.write(f"[RF] OOF AUC={overall['auc']:.4f} F1={overall['f1']:.4f} Thr~={avg_thr:.3f}")
    save_roc_plot(y_all.to_numpy(), oof_proba, figs_dir / "rf_oof_roc.png", "RF OOF ROC")

    # Refit on full data
    final_pipe = Pipeline([("pre", pre), ("clf", rf)])
    tqdm.write("[RF] Refit on full data...")
    final_pipe.fit(X_all, y_all)
    joblib.dump(final_pipe, models_dir / "rf_cv.joblib")

    # Export feature importances (align with transformed feature names)
    # cat names from OneHot
    if kept_cat:
        ohe = final_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(kept_cat).tolist()
    else:
        cat_out = []

    # numeric names after VarianceThreshold
    if kept_num:
        num_varth = final_pipe.named_steps["pre"].named_transformers_["num"].named_steps["varth"]
        support = num_varth.get_support()
        num_out = [c for c, keep in zip(kept_num, support) if keep]
    else:
        num_out = []

    feat_names = np.array(num_out + cat_out)

    # In scikit-learn, RF's feature_importances_ match the transformed matrix columns
    importances = final_pipe.named_steps["clf"].feature_importances_
    min_len = min(len(feat_names), len(importances))
    feat_names = feat_names[:min_len]
    importances = importances[:min_len]
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    imp_df.to_csv(reports_dir / "rf_feature_importances.csv", index=False)

    # Save metrics
    pd.DataFrame(fold_rows).to_excel(metrics_dir / "rf_folds.xlsx", index=False)
    pd.DataFrame([overall]).to_excel(metrics_dir / "rf_overall.xlsx", index=False)
    with open(metrics_dir / "rf_summary.json", "w") as f:
        json.dump({
            "oof_auc": overall["auc"],
            "oof_f1": overall["f1"],
            "avg_threshold": avg_thr,
            "kept_num": kept_num,
            "kept_cat": kept_cat,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": ("balanced" if args.class_weight_balanced else None),
        }, f, indent=2)

    tqdm.write(f"[RF] Saved model -> {models_dir/'rf_cv.joblib'}")
    tqdm.write(f"[RF] Metrics & plots -> {reports_dir}/(metrics|figures)")


if __name__ == "__main__":
    main()