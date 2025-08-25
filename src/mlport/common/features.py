import pandas as pd
from typing import Tuple, List

IDCOLS = ['SK_ID_CURR']

TARGET = 'TARGET'

CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

# Splitter
def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Splits a cleaned DataFrame into:
        - X (features only, no ID or target)
        - y (target vector), or None if Target not in df)
        - num_cols (list of numeric feature columns)
        - cat_cols (list of categorical feature columns)
    
    Args:
        df: a DataFrame after cleaning/merging

    Returns:
        X: features DataFrame
        y: Series (or None if TARGET missing)
        num_cols: list of numeric feature column names
        cat_cols: list of categorical feature column names
    """

    # Defensive copy
    df = df.copy()

    # y if present
    y = df[TARGET].astype(int) if TARGET in df.columns else None

    # Drop ID + TARGET to form feature matrix
    drop_cols = [c for c in IDCOLS if c in df.columns]
    if TARGET in df.columns:
        drop_cols.append(TARGET)
    X = df.drop(columns=drop_cols)

    # Split into numeric vs categorical
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X.columns if c in CATEGORICAL]

    return X, y, num_cols, cat_cols