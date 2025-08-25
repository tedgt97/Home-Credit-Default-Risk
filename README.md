#ML Portfolio
My data science portfolio project.


Stage_1.5-ML-Portfolio/
│
├── configs/                     # YAML configs for data paths, model params
│   └── clean.yaml
│
├── data/                        # Data storage (ignored by git)
│   ├── raw/                     # Kaggle original CSVs
│   └── processed/               # Cleaned / Stage 1.5 / splits
│
├── models/                      # Trained model binaries (.joblib / .pkl)
│
├── reports/                     # Outputs
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── peeks/
│
├── notebooks/                   # EDA & storytelling notebooks
│   ├── 01_eda_import_clean.ipynb
│   ├── 02_eda_stage15.ipynb
│   └── 03_conclusion.ipynb
│
├── src/mlport/                  # Your Python package ("mlport")
│   ├── __init__.py
│   │
│   ├── common/                  # reusable helpers
│   │   ├── __init__.py
│   │   ├── data.py              # load/save, Kaggle download
│   │   ├── clean.py             # cleaning logic
│   │   ├── merge.py             # merging logic
│   │   ├── features.py          # Stage 1.5 feature engineering
│   │   ├── metrics.py           # evaluation metrics (AUC, KS, etc.)
│   │   └── plots.py             # plotting utilities
│   │
│   ├── pipelines/               # orchestration scripts
│   │   ├── __init__.py
│   │   └── clean_merge.py       # end-to-end: download → clean → merge → save
│   │
│   ├── logistic/                # logistic regression code
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── random_forest/           # random forest code
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── clustering/              # clustering code
│       ├── __init__.py
│       ├── train.py
│       └── evaluate.py
│
├── tests/                       # pytest tests
│   └── test_metrics.py
│
├── requirements.txt             # dependencies
├── README.md                    # project overview & how to run
└── .gitignore
