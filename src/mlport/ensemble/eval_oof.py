import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def metrics_for(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"auc": auc, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": float(thr)}

def save_roc(y_true, y_prob, out, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Evaluate L1/L2 ensembling from OOF predictions")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    metrics_dir = reports / "metrics"
    figs_dir = reports / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---- load OOF probs ----
    lr_oof_path = metrics_dir / "logreg_oof_probs.csv"
    rf_oof_path = metrics_dir / "rf_oof_probs.csv"
    if not lr_oof_path.exists() or not rf_oof_path.exists():
        raise FileNotFoundError("Missing OOF files. Expected logreg_oof_probs.csv and rf_oof_probs.csv in reports/metrics/. Re-run trainers.")

    df_lr = pd.read_csv(lr_oof_path)
    df_rf = pd.read_csv(rf_oof_path)

    # join on idx and sanity check
    df = pd.merge(df_lr, df_rf, on=["idx","y_true"], how="inner")
    y = df["y_true"].to_numpy()
    p_lr = df["y_prob_logreg"].to_numpy()
    p_rf = df["y_prob_rf"].to_numpy()

    # ---- load thresholds from summaries ----
    lr_sum = load_json(metrics_dir / "logreg_summary.json")
    rf_sum = load_json(metrics_dir / "rf_summary.json")
    t_lr = float(lr_sum.get("avg_threshold", 0.5))
    t_rf = float(rf_sum.get("avg_threshold", 0.5))

    # ---- base model metrics ----
    m_lr = metrics_for(y, p_lr, t_lr)
    m_rf = metrics_for(y, p_rf, t_rf)

    # ---- ensembles ----
    # Hard rules:
    # L1 = "at least one says default" = OR of hard predictions → pred = (p_lr>=t_lr) OR (p_rf>=t_rf)
    # L2 = "at least two say default"  ; with 2 models, that means BOTH default → AND
    yhat_lr = (p_lr >= t_lr).astype(int)
    yhat_rf = (p_rf >= t_rf).astype(int)
    yhat_L1 = ((yhat_lr + yhat_rf) >= 1).astype(int)
    yhat_L2 = ((yhat_lr + yhat_rf) >= 2).astype(int)

    # Soft scores for ROC:
    # For L1 (OR), a common soft proxy is max(probabilities)
    # For L2 (AND), a conservative soft proxy is min(probabilities)
    p_L1 = np.maximum(p_lr, p_rf)
    p_L2 = np.minimum(p_lr, p_rf)

    # thresholds for L1/L2 to match prevalence (use same prevalence as y)
    prev = y.mean()
    t_L1 = float(np.quantile(p_L1, 1 - prev))
    t_L2 = float(np.quantile(p_L2, 1 - prev))

    m_L1 = metrics_for(y, p_L1, t_L1)
    m_L2 = metrics_for(y, p_L2, t_L2)

    # ---- save ROC plots ----
    save_roc(y, p_lr, figs_dir / "ensemble_logreg_oof_roc.png", "LogReg OOF ROC")
    save_roc(y, p_rf, figs_dir / "ensemble_rf_oof_roc.png", "RF OOF ROC")
    save_roc(y, p_L1, figs_dir / "ensemble_L1_oof_roc.png", "L1 (OR) OOF ROC")
    save_roc(y, p_L2, figs_dir / "ensemble_L2_oof_roc.png", "L2 (AND) OOF ROC")

    # ---- write metrics tables ----
    rows = []
    rows.append({"model":"M1_logreg", **m_lr})
    rows.append({"model":"M2_rf", **m_rf})
    rows.append({"model":"L1_or", **m_L1})
    rows.append({"model":"L2_and", **m_L2})
    out_xlsx = metrics_dir / "ensemble_summary.xlsx"
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)

    # ---- comparison bar chart (AUC) ----
    auc_df = pd.DataFrame({"model":["M1_logreg","M2_rf","L1_or","L2_and"],
                           "auc":[m_lr["auc"], m_rf["auc"], m_L1["auc"], m_L2["auc"]]})
    plt.figure()
    plt.bar(auc_df["model"], auc_df["auc"])
    plt.xticks(rotation=20)
    plt.ylabel("AUC")
    plt.title("AUC Comparison: M1, M2, L1, L2")
    plt.ylim(0.0, 1.0)
    plt.savefig(figs_dir / "ensemble_auc_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    # ---- also save a CSV with the hard predictions (useful for debugging) ----
    pred_df = pd.DataFrame({
        "idx": df["idx"],
        "y_true": y,
        "p_logreg": p_lr,
        "p_rf": p_rf,
        "p_L1": p_L1,
        "p_L2": p_L2,
        "yhat_logreg": yhat_lr,
        "yhat_rf": yhat_rf,
        "yhat_L1": yhat_L1,
        "yhat_L2": yhat_L2,
    })
    pred_df.to_csv(metrics_dir / "ensemble_oof_preds.csv", index=False)

    print(f"[ENSEMBLE] Wrote metrics → {out_xlsx}")
    print(f"[ENSEMBLE] Plots → {figs_dir}/ensemble_*")
    print(f"[ENSEMBLE] Predictions → {metrics_dir/'ensemble_oof_preds.csv'}")

if __name__ == "__main__":
    main()
