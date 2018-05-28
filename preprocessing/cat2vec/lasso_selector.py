"""
The objective of this script is to select the best categories of a high
cardinality categorical feature using LASSO penalization.

>> reload_ext autoreload
>> autoreload 2
"""
import pandas as pd
import numpy as np
import category_encoders as ce

from preprocessing.utils import preprocess
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy import sparse


def lasso_coeficients(path, features, target):
    """
    computes a classification task using ols LASSO algorithm and returns the
    coefficients.
    L1 penalization is used to get sparse coefficients.

    Parameters
    ----------
    path : str
        path to fetch the dataframe containing features and targets

    features : str
        name of the feature column

    target : str
        name of the target columns

    Returns
    -------
    array, shape (n_features) | (n_classes, n_features)

    """
    df = pd.read_csv(path, sep=';')
    X = df[features]
    y = df[target]

    X = preprocess.normalize_cat(X, 'strings')
    X_dummies = ce.OneHotEncoder().fit_transform(X.values)

    X_sparse = sparse.csr_matrix(X_dummies.values)

    param_grid = {'alpha': np.logspace(-6, 2, num=10),
                  'max_iter': np.linspace(100, 10000, num=10),
                  'tol': np.logspace(-6, 1, num=10)}

    scoring = {'AUC': 'roc_auc', 'Precision': 'precision'}

    grid = GridSearchCV(Lasso(), param_grid, scoring=scoring, n_jobs=-1,
                        cv=5, verbose=3)

    grid.fit(X_sparse, y)

    print(grid.cv_results_)

    print('Best score for LASSO: {} \n obtained with following '
          'parameters: {}'.format(grid.best_score_, grid.best_params_))

    lasso = grid.best_estimator_

    lasso.fit(X_sparse, y)

    return lasso.coef_
    
