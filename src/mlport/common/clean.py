import pandas as pd
import numpy as np

# Application cleaning

def clean_application_base(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()

    # String columns lower cleaning
    for c in df.select_dtypes(include='object').columns:
        df[c] = (df[c].astype("string").str.strip().str.lower())
    
    