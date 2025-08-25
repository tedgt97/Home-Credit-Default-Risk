from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from mlport.common.data import load_any
from mlport.common.features import IDCOLS  # ["SK_ID_CURR"]

def score_dataset(model_path: str | Path, data_path: str | Path, proba_col: str):
    """
    Load a fitted sklearn pipeline (.joblib) and score a parquet/csv.
    Returns (df_ids, proba) where df_ids has IDCOLS.
    """
    pipe = joblib.load(model_path)
    df = load_any(data_path)

    # keep ids separately for joining to output
    id_df = df[[c for c in IDCOLS if c in df.columns]].copy()
    X = df.drop(columns=[c for c in IDCOLS if c in df.columns], errors="ignore")  # TARGET not in test

    # predict_proba -> probabilities for positive class at [:,1]
    proba = pipe.predict_proba(X)[:, 1]
    out = id_df.copy()
    out[proba_col] = proba
    return out

def read_avg_threshold(summary_json: str | Path, default=0.5) -> float:
    p = Path(summary_json)
    if not p.exists():
        return float(default)
    with open(p, "r") as f:
        js = json.load(f)
    return float(js.get("avg_threshold", default))
