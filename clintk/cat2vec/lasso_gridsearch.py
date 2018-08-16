"""
The objective of this script is to select the best categories of a high
cardinality categorical feature using LASSO penalization.

For the moment only binary/continuous logistic regression is implemented

>> reload_ext autoreload
>> autoreload 2
"""
import pandas as pd
import numpy as np

from preprocessing.cat2vec import tools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def lr_coefficients(path, features, targets, key, output_path, **kwargs):
    """
    Performs categorical variable selection using L1-penalized logistic
    regression model

    It only supports binary or continuous target for the moment


    Parameters
    ----------
    path : str
        input path or url for the dataframe

    features : str
        column name of the categorical column

    targets : str
        name of the target column in the df

    key : str
        key to group categorical variables

    output_path : str
        path to save the coefficients in a csv file

    kwargs
        keyword arguments for the hyperparameter grid

    Returns
    -------
    array
    the coefficients of the L1-logistic regression

    Examples
    --------
    >>> lr_coefficients('input.csv', 'medication_name', 'target', \
    solver=['liblinear', 'saga'], C=np.logspace(-6, 2, 10))

    """
    df = pd.read_csv(path, sep=';')
    df[features] = tools.normalize_cat(df[features], 'strings')
    dummies = pd.get_dummies(df[[key, features, targets]],
                             columns=[features])

    # avoid target replication
    agg_dic = {targets: 'first'}
    for colname in list(dummies.columns)[2:]:
        # summing dummy variables to have more than one 1 on each row
        agg_dic[colname] = 'sum'

    dummies_group = dummies.groupby(by=key, as_index=False).agg(agg_dic)

    y = dummies_group[targets]
    X = dummies_group.iloc[:, 2:]

    param_grid = {}
    for key, value in kwargs.items():
        param_grid[key] = value

    # using metrics for imbalanced dataset
    scoring = {'AUC': 'roc_auc', 'Precision': 'precision'}
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)

    grid = GridSearchCV(LogisticRegression(penalty='l1', n_jobs=8),
                        param_grid=param_grid, scoring=scoring,
                        cv=cv, refit='AUC',
                        n_jobs=8, verbose=5)

    grid.fit(X, y)

    print(grid.cv_results_)

    print('Best score for LASSO: {} \n obtained with following '
          'parameters: {}'.format(grid.best_score_, grid.best_params_))

    lr = grid.best_estimator_

    colnames = np.array(dummies_group.columns)[2:]
    df_coefs = pd.DataFrame({'features': colnames,
                             'coef': lr.coef_.ravel()})

    df_coefs.to_csv(output_path,
                    sep=';',
                    encoding='utf-8')

    return lr.coef_
