from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
import pandas as pd

def load_any(path: str) -> pd.DataFrame:
    p = str(path).lower()
    return pd.read_parquet(path) if p.endswith('.parquet') else pd.read_csv(path)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

# Kaggle

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _unzip_all(zip_dir: str | Path, out_dir: str | Path) -> None:
    zip_dir, out_dir = Path(zip_dir), _ensure_dir(out_dir)
    for z in zip_dir.glob('*.zip'):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_dir)

def download_kaggle_competition(competition: str, out_dir: str | Path, unzip: bool = True) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Kaggle package is not installed. Please install it with `pip install kaggle`.") from e
    
    out_dir = _ensure_dir(out_dir)
    
    api = KaggleApi()
    api.authenticate()

    # if file is already downloaded
    already = any(Path(out_dir).glob('*.zip')) or any(Path(out_dir).glob('*.csv'))
    if not already:
        api.competition_download_files(competition=competition, path=str(out_dir), quiet=False)
    
    if unzip:
        _unzip_all(out_dir, out_dir)

# CLI entrypoint

def _cli():
    ap = argparse.ArgumentParser(description="Data utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("kaggle", help="Download Kaggle competition data.")
    d.add_argument("--competition", required=True, help="e.g., home-credit-default-risk")
    d.add_argument("--out", default="data/raw", help="directory to save files")
    d.add_argument("--no-unzip", action="store_true", help="skip unzipping")
    args = ap.parse_args()

    if args.cmd == "kaggle":
        download_kaggle_competition(args.competition, args.out, unzip=(not args.no_unzip))

if __name__ == "__main__":
    _cli()

