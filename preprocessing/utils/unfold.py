"""
unfolds merges dataframes into a big feature matrix 
All the features are labeled with a date and two keys for identification


"""
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


def unfold(df, key1, key2, feature, value, date):
    """
    Takes a dataframe[key1, key2, feature, value, date] to build a matrix of
    the parameters grouped by [key1, key2, key3]

    Parameters
    ----------
    df
    id1
    id2
    feature
    value
    date
    target

    Returns
    -------

    """
    df_res = df.loc[:, [key1, key2, date]]

    # dictionnary for aggregation
    agg_dic = {}

    for feature_name in df[feature].unique():
        agg_dic[feature_name] = 'mean'

        new_col = []
        for i in df.index:
            if df.at[i, feature] == feature_name:
                new_col.append(df.at[i, value])
            else:
                new_col.append(np.nan)

        df_res.loc[:, feature_name] = new_col

    df_grouped = df_res.groupby(by=[key1, key2, date], sort=False,
                                as_index=False).agg(agg_dic)

    return df_grouped


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

    for i in range(new_col.shape[0]):
        for j in range(new_col.shape[1]):
            row = {key1: df.at[i, key1],
                   key2: df.at[i, key2],
                   feature: df.at[i, feature] + '_{}'.format(j),
                   value: new_col[i, j],
                   date: df.at[i, date]}

            df_res = df_res.append(row, ignore_index=True)

    return df_res





