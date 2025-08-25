import argparse
from pathlib import Path
from mlport.common.score import score_dataset, read_avg_threshold

def main():
    ap = argparse.ArgumentParser(description="Score test with Random Forest model")
    ap.add_argument("--test", default="data/processed/test.parquet")
    ap.add_argument("--model", default="models/rf_cv.joblib")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    subs_dir = reports / "submissions"; subs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports / "metrics"

    out = score_dataset(args.model, args.test, proba_col="TARGET_PROB_rf")

    thr = read_avg_threshold(metrics_dir / "rf_summary.json", default=0.5)
    out["PRED_rf"] = (out["TARGET_PROB_rf"] >= thr).astype(int)

    out_path = subs_dir / "pred_rf_test.csv"
    out.to_csv(out_path, index=False)
    print(f"[RF] Wrote {out_path}")

if __name__ == "__main__":
    main()
