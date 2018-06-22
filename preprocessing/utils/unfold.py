"""
unfolds merges dataframes into a big feature matrix 
All the features are labeled with a date and two keys for identification


"""
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from multiprocessing.pool import Pool
from sklearn.base import BaseEstimator


class Unfolder(BaseEstimator):
    """
    Takes a dataframe[key1, key2, feature, value, date] to build a matrix of
    the parameters grouped by [key1, key2, key3]

    Parameters
    ----------
    df
    key1
    key2
    feature
    value
    date

    Returns
    -------

    """
    def __init__(self, key1, key2, feature, value, date, n_jobs=1):
        self.key1 = key1
        self.key2 = key2
        self.feature = feature
        self.value = value
        self.date = date
        self.n_jobs = n_jobs

    def fit(self, df, y=None):
        # stocking df and agg_dic for multipocessing convenience
        self.df_ = df

    def unfold(self):
        df_res = self.df_.loc[:, [self.key1, self.key2, self.date]]

        if self.n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(self.n_jobs)

        unique_features = self.df_[self.feature].unique()
        new_cols = pool.map(self.add_columns, unique_features)

        df_res.reset_index(drop=True, inplace=True)
        df_res = pd.concat([df_res] + [new_col for new_col in new_cols],
                           axis=1)

        # agregation function for group by
        agg_dic = {key: 'mean' for key in unique_features}
        df_grouped = df_res.groupby(by=[self.key1, self.key2, self.date],
                                    sort=False,
                                    as_index=False).agg(agg_dic)
        return df_grouped

    def add_columns(self, feature_name):
        new_col = []
        for i in self.df_.index:
            if self.df_.at[i, self.feature] == feature_name:
                new_col.append(self.df_.at[i, self.value])
            else:
                new_col.append(np.nan)

        return pd.DataFrame(new_col, columns=[feature_name])



def transform_and_label(df, key1, key2, date,  feature, value,
                        estimator, **kwargs):
    """ Takes dataframe as input, applies transformation on value column and
    returns  df with a new columns of the transformed feature

    The transformation returns a copy of the input dataframe

    Only implements unsupervised transformation

    Parameters
    ----------
    df : pandas.DataFrame
        should contain only one unique value in its `feature` column

    feature : str
        features names column

    value : str
        features values column


    estimator : sklearn.BaseEstimator
        sklearn compatible transformer that implements .fit() and
        .transform() methods

    **kwargs : additional keyword arguments for estimator object

    Returns
    -------
    pandas.DataFrame
        same as df with additional rows  for the transformed feature

    """
    # unsupervised transformation
    old_col = df[value].values
    transformer = estimator(**kwargs).fit(old_col)

    new_col = transformer.transform(old_col)
    
    # converts to numpy.ndarray
    if type(new_col) in [pd.DataFrame, pd.Series]:
        new_col = new_col.values
    elif type(new_col) == csr_matrix:
        new_col = new_col.todense()
    else:
        new_col = new_col

    df_res = pd.DataFrame(None, columns=df.columns)

    # filling new rows
    for i in range(new_col.shape[0]):
        for j in range(new_col.shape[1]):
            row = {key1: df.at[i, key1],
                   key2: df.at[i, key2],
                   feature: df.at[i, feature] + '_{}'.format(j),
                   value: new_col[i, j],
                   date: df.at[i, date]}

            df_res = df_res.append(row, ignore_index=True)

    return df_res





