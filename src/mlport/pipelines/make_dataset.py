import argparse
from pathlib import Path
import pandas as pd
from mlport.common.data import load_any, save_parquet
from mlport.common.merge import merge_all_features
from mlport.common.clean import (clean_for_model, create_computed, normalize_categoricals, find_issue_frames, filter_out_issues)

def build_and_save(raw_dir: Path, out_dir: Path) -> None:
    """
    Build train/test datasets by merging + cleaning raw Home Credit files.
    Save results as Parquet under out_dir.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Raw
    app_train = load_any(raw_dir / "application_train.csv")
    app_test = load_any(raw_dir / "application_test.csv")
    bureau = load_any(raw_dir / "bureau.csv")
    prev = load_any(raw_dir / "previous_application.csv")
    inst = load_any(raw_dir / "installments_payments.csv")

    print(f"[INFO] Loaded raw files: "
          f"train={app_train.shape}, test={app_test.shape}", 
          f"bureau={bureau.shape}, prev={prev.shape}, inst={inst.shape}")
    
    # Merge features
    train_merged = merge_all_features(app_train, bureau, prev, inst)
    test_merged = merge_all_features(app_test, bureau, prev, inst)

    print(f"[INFO] Merged features: "
          f"train={train_merged.shape}, test={test_merged.shape}")
    
    # Cleaning
    train_clean = clean_for_model(train_merged)
    test_clean = clean_for_model(test_merged)

    print(f"[INFO] Cleaned datasets: "
          f"train={train_clean.shape}, test={test_clean.shape}")
    
    # Audit Issues
    issues = find_issue_frames(train_clean)
    # Build Issue Summary
    summary_rows = []
    total_rows = len(train_clean)
    for name, df_issue in issues.items():
        count = len(df_issue)
        rate = round(100 * count / total_rows, 2) if total_rows else 0.0
        summary_rows.append({'issue': name, 'count': count, 'rate_percent': rate})
    summary_df = pd.DataFrame(summary_rows)

    # Write Issues to Excel
    issues_xlsx = out_dir / "train_issues.xlsx"
    with pd.ExcelWriter(issues_xlsx, engine='openpyxl') as xw:
        summary_df.to_excel(xw, sheet_name="Summary", index=False)
        for name, df_issues in issues.items():
            sheet = name[:31]
            df_issues.to_excel(xw, sheet_name=sheet, index=False)

    print(f"[INFO] Wrote issue audit to {issues_xlsx}")
    
    # Filter Issues out of train
    train_clean = filter_out_issues(train_clean)

    print(f"[INFO] Filter Out training: "
          f"train={train_clean.shape}, test={test_clean.shape}")

    # Save to Parquet
    save_parquet(train_clean, out_dir / "train.parquet")
    save_parquet(test_clean,  out_dir / "test.parquet")

    print(f"[INFO] Saved processed datasets to {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Build processed datasets")
    ap.add_argument("--raw_dir", default="data/raw", help="Directory with raw CSVs")
    ap.add_argument("--out_dir", default="data/processed", help="Directory to save processed Parquet files")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    build_and_save(raw_dir, out_dir)


if __name__ == "__main__":
    main()