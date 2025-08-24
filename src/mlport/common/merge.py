import pandas as pd

# Bureau
def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:

    g = bureau.groupby('SK_ID_CURR', dropna=True).agg({
        'SK_ID_BUREAU': 'count',
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_SUM_OVERDUE': 'sum',
        'CREDIT_TYPE': 'nunique'
    }).reset_index()
    g.columns = ['SK_ID_CURR'] + [f'BUREAU_{col[0]}_{col[1].upper()}' for col in g.columns[1:]]
    
    return g

# Previous Application
def aggregate_previous_app(prev: pd.DataFrame) -> pd.DataFrame:

    df = prev.copy()
    if {'AMT_APPLICATION', 'AMT_CREDIT'}.issubset(df.columns):
        df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']

    g = df.groupby('SK_ID_CURR', dropna=True).agg({
        'SK_ID_PREV': 'count',
        'AMT_ANNUITY': ['mean', 'max'],
        'AMT_APPLICATION': ['mean', 'max'],
        'APP_CREDIT_PERC': ['mean', 'max', 'min']
    }).reset_index()
    g.columns = ['SK_ID_CURR'] + [f'PREV_APP_{col[0]}_{col[1].upper()}' for col in g.columns[1:]]

    return g

# Installments Payments
def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:

    df = installments.copy()
    if {'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT'}.issubset(df.columns):
        df['DPD'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
        df['DBD'] = (df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    if {'AMT_PAYMENT', 'AMT_INSTALMENT'}.issubset(df.columns):
        df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    
    g = df.groupby('SK_ID_CURR', dropna=True).agg({
        'DPD': ['mean', 'max', 'sum'],
        'DBD': ['mean', 'max', 'sum'],
        'PAYMENT_PERC': ['mean', 'max', 'min'],
        'PAYMENT_DIFF': ['mean', 'max', 'sum']
    }).reset_index()
    g.columns = ['SK_ID_CURR'] + [f'INSTALLMENTS_{col[0]}_{col[1].upper()}' for col in g.columns[1:]]

    return g

# Merge complete
def merge_all_features(
        app: pd.DataFrame,
        bureau: pd.DataFrame,
        previous_app: pd.DataFrame,
        installments: pd.DataFrame
) -> pd.DataFrame:
    
    bur_agg = aggregate_bureau(bureau)
    prev_agg = aggregate_previous_app(previous_app)
    inst_agg = aggregate_installments(installments)

    out = app.merge(bur_agg, on='SK_ID_CURR', how='left')
    out = out.merge(prev_agg, on='SK_ID_CURR', how='left')
    out = out.merge(inst_agg, on='SK_ID_CURR', how='left')
    
    return out

