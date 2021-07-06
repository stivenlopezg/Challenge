import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            raise TypeError("Error: el parametro columns debe ser una lista.")
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns]
        return X


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            raise TypeError("Error: el parametro columns debe ser una lista.")
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.get_dummies(data=pd.DataFrame(X, columns=self.columns), columns=self.columns)
        elif isinstance(X, pd.DataFrame):
            X = pd.get_dummies(data=X, columns=self.columns)
        else:
            raise TypeError("Error: El parametro X debe ser un pd.DataFrame o np.ndarray.")
        return X


class GetDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            raise TypeError("Error: el parametro columns debe ser una lista.")
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
        elif isinstance(X, pd.DataFrame):
            pass
        else:
            raise TypeError("Error: El parametro X debe ser un pd.DataFrame o np.ndarray.")
        return X

