"""
unfolds merges dataframes into a big feature matrix 
All the features are labeled with a date and two keys for identification

Better explainations and schemas can be found on the repo wiki
"""
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from multiprocessing.pool import Pool
from sklearn.base import BaseEstimator


class Unfolder(BaseEstimator):
    """
    Takes a dataframe[key1, key2, feature, value, date] to build a matrix of
    the parameters grouped by [key1, key2, date]

    This object is to be used after a timeframe of the feature has been build to
    group them into a feature matrix.
    The idea is to facilitate the data preparation for a sequential learning
    task.

    Parameters
    ----------
    key1 : str
        primary key

    key2 : str
        secondary key

    feature : str
        name of the feature

    value : float
        value of the feature `feature`

    date : datetime
        date at which `feature` was measured

    n_jobs : int
        number of CPUs to use for computation. If -1, all the available cores
        are used

    """
    def __init__(self, key1, key2, feature, value, date, n_jobs=1):
        self.key1 = key1
        self.key2 = key2
        self.feature = feature
        self.value = value
        self.date = date
        self.n_jobs = n_jobs
        self.df_ = None

    def fit(self, df):
        """  saves dataframe for multiprocessing convenience

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        self

        """
        self.df_ = df
        return self


    def unfold(self):
        """ performs the unfolding transformation

        Returns
        -------
        pandas.DataFrame
            The dataframe that contains the added feature columns
            Rows are ordered by [key1, key2, date] for convenience

        """
        df_res = self.df_.loc[:, [self.key1, self.key2, self.date]]

        if self.n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(self.n_jobs)

        unique_features = self.df_[self.feature].unique()
        new_cols = pool.map(self.add_columns, unique_features)

        pool.close()
        pool.join()

        df_res.reset_index(drop=True, inplace=True)
        df_res = pd.concat([df_res] + [new_col for new_col in new_cols],
                           axis=1)

        # aggregation function for group by
        agg_dic = {key: 'mean' for key in unique_features}
        df_grouped = df_res.groupby(by=[self.key1, self.key2, self.date],
                                    sort=False,
                                    as_index=False).agg(agg_dic)
        return df_grouped

    def add_columns(self, feature_name):
        """ adds a column of a given feature

        This auxiliary function is to ease the use of multiprocess.pool.Pool

        Parameters
        ----------
        feature_name : str
            name of the feature we are adding to the dataframe

        Returns
        -------
        pandas.DataFrame
            contains a single column `feature_name` that contains values
            or NaN depending on the presence of the feature for each row

        """
        new_col = []
        for i in self.df_.index:
            if self.df_.at[i, self.feature] == feature_name:
                new_col.append(self.df_.at[i, self.value])
            else:
                new_col.append(np.nan)

        return pd.DataFrame(new_col, columns=[feature_name])


def transform_and_label(df, key1, key2, date,  feature, value,
                        estimator, return_estimator=False,
                        **kwargs):
    """ Takes dataframe as input, applies transformation on value column and
    returns  df with a new columns of the transformed feature

    The transformation returns a copy of the input dataframe

    Only implements unsupervised transformation

    @TODO keep sparse representation to unfold data

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

    return_estimator : bool
        if true, returns the trained estimator

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
    # elif type(new_col) == csr_matrix:
    #     new_col = new_col.todense()
    else:
        new_col = new_col

    df_res = pd.DataFrame(None, columns=df.columns)

    # filling new rows
    if type(new_col) == csr_matrix:
        rows, cols = new_col.nonzero()
        for i, j in zip(rows, cols):
            row = {key1: df.at[i, key1],
                   key2: df.at[i, key2],
                   feature: df.at[i, feature] + '_{}'.format(j),
                   value: new_col[i, j],
                   date: df.at[i, date]}

            df_res = df_res.append(row, ignore_index=True)

    else:
        for i in range(new_col.shape[0]):
            for j in range(new_col.shape[1]):
                row = {key1: df.at[i, key1],
                       key2: df.at[i, key2],
                       feature: df.at[i, feature] + '_{}'.format(j),
                       value: new_col[i, j],
                       date: df.at[i, date]}

                df_res = df_res.append(row, ignore_index=True)

    if return_estimator:
        return df_res, transformer
    else:
        return df_res
