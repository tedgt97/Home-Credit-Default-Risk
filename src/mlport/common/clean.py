import pandas as pd

SELECTED = [
    "SK_ID_CURR", "TARGET",
    "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "NAME_INCOME_TYPE", "DAYS_BIRTH",
    "c_EXT_SOURCE", 'BUREAU_SK_ID_BUREAU_COUNT', 'BUREAU_CREDIT_DAY_OVERDUE_MEAN', 'BUREAU_CREDIT_DAY_OVERDUE_MAX', 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM', 'BUREAU_CREDIT_TYPE_NUNIQUE', 'PREV_APP_SK_ID_PREV_COUNT', 'PREV_APP_AMT_ANNUITY_MEAN', 'PREV_APP_AMT_ANNUITY_MAX', 'PREV_APP_AMT_APPLICATION_MEAN', 'PREV_APP_AMT_APPLICATION_MAX', 'PREV_APP_APP_CREDIT_PERC_MEAN', 'PREV_APP_APP_CREDIT_PERC_MAX', 'PREV_APP_APP_CREDIT_PERC_MIN', 'INSTALLMENTS_DPD_MEAN', 'INSTALLMENTS_DPD_MAX', 'INSTALLMENTS_DPD_SUM',
     'INSTALLMENTS_DBD_MEAN', 'INSTALLMENTS_DBD_MAX', 'INSTALLMENTS_DBD_SUM', 'INSTALLMENTS_PAYMENT_PERC_MEAN', 'INSTALLMENTS_PAYMENT_PERC_MAX', 'INSTALLMENTS_PAYMENT_PERC_MIN', 'INSTALLMENTS_PAYMENT_DIFF_MEAN', 'INSTALLMENTS_PAYMENT_DIFF_MAX', 'INSTALLMENTS_PAYMENT_DIFF_SUM'
    ]

CRITICAL = [
    "TARGET", "c_EXT_SOURCE"
]

CATEGORICAL = [
    "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE"
]

def create_computed(df: pd.DataFrame) -> pd.DataFrame:
    # c_EXT_SOURCE: coalescing EXT_SOURCE 1/2/3 if missing
    if 'c_EXT_SOURCE' not in df.columns and {'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'}.issubset(df.columns):
        df = df.copy()
        df['c_EXT_SOURCE'] = df['EXT_SOURCE_1'].fillna(
            df['EXT_SOURCE_2'].fillna(
                df['EXT_SOURCE_3']
            )
        )
    return df

def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categoricals to lowercase for consistent encoding later.
    """
    df = df.copy()
    for c in (set(CATEGORICAL) & set(df.columns)):
        df[c] = df[c].astype("string").str.strip().str.lower()
    return df

def drop_missing_required(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing required columns (TARGET, c_EXT_SOURCE).
    """
    present = [c for c in CRITICAL if c in df.columns]
    if not present:
        return df.copy()
    return df.dropna(subset=present).copy()

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the final selected columns.
    """
    keep = [c for c in SELECTED if c in df.columns]
    return df[keep].copy()

def clean_for_model(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Clean & reduce merged dataset for modeling:
        - create c_EXT_SOURCE
        - normalize categoricals
        - drop rows missing TARGET or c_EXT_SOURCE
        - keep only selected columns
    """
    df = create_computed(merged)
    df = normalize_categoricals(df)
    df = drop_missing_required(df)
    df = select_columns(df)

    return df

# ISSUE Audit
ISSUE_DEFS = {
    "missing_target": lambda df: df['Target'].isna() if 'Target' in df.columns else pd.Series(False, index=df.index),
    "missing_c_ext_source": lambda df: df['c_EXT_SOURCE'].isna() if 'c_EXT_SOURCE' in df.columns else pd.Series(False, index=df.index),
    "missing_bureau_overdue": lambda df: df['BUREAU_CREDIT_DAY_OVERDUE_MEAN'].isna() if 'BUREAU_CREDIT_DAY_OVERDUE_MEAN' in df.columns else pd.Series(False, index=df.index),
    "missing_installments_dpd": lambda df: df['INSTALLMENTS_DPD_MEAN'].isna() if 'INSTALLMENTS_DPD_MEAN' in df.columns else pd.Series(False, index=df.index),
}

def find_issue_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Return a dict of {issue_name: DataFrame of rows triggering that issue".
    Issues may overlap.
    """
    issues = {}
    n = len(df)
    for name, rule in ISSUE_DEFS.items():
        mask = rule(df)
        if mask.any():
            issues[name] = df.loc[mask].copy()
        else:
            # stil include empty sheet for completeness
            issues[name] = df.iloc[0:0].copy()

    return issues

def filter_out_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows that trigger ANY issue, mirroring the notebook logic.
    """
    if df.empty:
        return df.copy()
    drop_mask = pd.Series(False, index=df.index)
    for rule in ISSUE_DEFS.values():
        drop_mask = drop_mask | rule(df)
    return df.loc[~drop_mask].copy()