import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer to convert +/-inf to NaN."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        return X
