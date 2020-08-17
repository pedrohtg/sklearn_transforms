from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class ZeroNorm(BaseEstimator, TransformerMixin):
    def __init__(self, columns, data, j_columns=None):
        self.mean = data[columns].mean()
        self.std = data[columns].std()
        self.columns = columns
        self.j_columns = j_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        d = (data[self.columns] - self.mean)/self.std
        if self.j_columns is not None:
            return d.join(X[self.j_columns]) 
        return d
