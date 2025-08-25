import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mlport.common.score import read_avg_threshold

def main():
    ap = argparse.ArgumentParser(description="Ensemble (L1/L2) predictions on test")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    subs_dir = reports / "submissions"; subs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports / "metrics"

    # inputs made by the per-model predict scripts
    p_lr = subs_dir / "pred_logreg_test.csv"
    p_rf = subs_dir / "pred_rf_test.csv"
    if not (p_lr.exists() and p_rf.exists()):
        raise FileNotFoundError("Run model predict scripts first: logreg.predict and random_forest.predict")

    df_lr = pd.read_csv(p_lr)   # SK_ID_CURR, TARGET_PROB_logreg, PRED_logreg
    df_rf = pd.read_csv(p_rf)   # SK_ID_CURR, TARGET_PROB_rf,     PRED_rf
    df = pd.merge(df_lr, df_rf, on="SK_ID_CURR", how="inner")

    # thresholds learned on train (used to recompute hard labels from probabilities)
    t_lr = read_avg_threshold(metrics_dir / "logreg_summary.json", default=0.5)
    t_rf = read_avg_threshold(metrics_dir / "rf_summary.json", default=0.5)

    # hard predictions from probabilities (reliable even if input PRED_* absent)
    yhat_lr = (df["TARGET_PROB_logreg"] >= t_lr).astype(int)
    yhat_rf = (df["TARGET_PROB_rf"]     >= t_rf).astype(int)

    # L1 = at least one model says default (OR)
    df["PRED_L1_atleast1"] = ((yhat_lr + yhat_rf) >= 1).astype(int)
    # L2 = both models say default (AND)
    df["PRED_L2_both"]     = ((yhat_lr + yhat_rf) >= 2).astype(int)

    # Soft proxies (single ensemble probability)
    df["TARGET_PROB_L1_atleast1"] = np.maximum(df["TARGET_PROB_logreg"], df["TARGET_PROB_rf"])
    df["TARGET_PROB_L2_both"]     = np.minimum(df["TARGET_PROB_logreg"], df["TARGET_PROB_rf"])

    out_cols = [
        "SK_ID_CURR",
        "TARGET_PROB_logreg", "PRED_logreg",
        "TARGET_PROB_rf",     "PRED_rf",
        "TARGET_PROB_L1_atleast1", "PRED_L1_atleast1",
        "TARGET_PROB_L2_both",     "PRED_L2_both",
    ]
    out = df[out_cols].copy()

    out_path = subs_dir / "pred_ensemble_test.csv"
    out.to_csv(out_path, index=False)
    print(f"[ENSEMBLE] Wrote {out_path}")

if __name__ == "__main__":
    main()
