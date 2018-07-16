"""
selects parameters with L1 logistic regression
"""
import pandas as pd

from sklearn.base import BaseEstimator


class LassoSelector(BaseEstimator):
    """
    This class is made to be used after cat2vec.lasso_gridsearch since it
    selects the features from a dataframe that have the most weighted
    coefficients (according to a L1-penalized linear model)

    It inherits from sklearn.base.BaseEstimator to allow gridsearching the
    best `n_features` using a pipeline and a basline classifier

    Parameters
    ----------
    n_features : int
        number of top features to keep

    lasso_coefs : pd.DataFrame
        each row is the name of a category and its coef weight in LASSO
        model

    feature_col : str
        name of the feature col (ie name of the categorical variable)

    coef_col : str
        name of the column of the LASSO coefficients in lasso_coefs dataframe

    Examples
    --------
    >>> dico = {'coef': [0, 4.5, 1.2, 0.3], \
                'colnames': ['feat1', 'feat2', 'feat3', 'feat4']}
    >>> df = pd.DataFrame(dico)
    keeps only feat2 and feat3
    >>> selector = LassoSelector(2).fit(df['colnames'], df['coef'])
    >>> X = [[0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]]
    >>> selector.transform(X)
    [[0, 1], [1, 0], [1, 0]]
    """
    def __init__(self, lasso_coefs, feature_col, coef_col,
                 n_features=64):
        self.n_features = n_features
        self.feature_col = feature_col
        self.lasso_coefs = lasso_coefs
        self.coef_col = coef_col

    def fit(self, X, y):
        return self

    def transform(self, X):
        """

        Parameters
        ----------
        X : pd.DataFrame
            contains only features

        Returns
        -------
        ndarray
            contains the best n_features
        """
        self.lasso_coefs['abs_coef'] = abs(self.lasso_coefs[self.coef_col])
        self.lasso_coefs.sort_values(['abs_coef'], ascending=False,
                                     inplace=True)

        # keeping top features according to lasso
        coefs_to_keep = self.lasso_coefs.iloc[:self.n_features, :]
        coefs_to_keep = coefs_to_keep[self.feature_col]

        return X[coefs_to_keep.values].values
