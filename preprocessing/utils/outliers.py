"""
Scripts to remove the outliers and na values from the different tables

To be used for the values that are mistyped

"""
import pandas as pd
import numpy as np


class OutlierRemover(object):
    """ removes outliers and replaces them by value given in dic_path or by 
    imputing the column mean value
    
    Parameters
    ----------
    dic_path: str   
        path to the dictionary containing outliers information
    
    inplace: bool, default=True
        True to perform the transformation inplace
        False to do it on a copy of the dataframe
        
    """
    def __init__(self, dic_path, inplace=True):
        self.path = dic_path
        self.inplace = inplace

    # for sklearn pipeline compatibility
    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return impute_df(X, self.path, self.inplace)


def impute_col(X, lbound, ubound, impute):
    """ imputes missing and mistyped values of one col of the dataframe

    Parameters
    ----------
    X : iterable, array-like
        column to which we want to impute missing values


        name of the column

    lbound : float
        lower bound for normal values

    ubound : float
        upper bound for normal values

    impute : float or None
        if float is given, replaces outlier by the given value
        if None, the mean value is returned


    Returns
    -------
    df.Series
        df.col_name except its wrong values are imputed according
        to strategy
    """
    impute_value = impute or np.mean(X)
    res = []

    for row in X:
        # check bounds + nan
        if (row < lbound) | (row > ubound) | (row == np.nan):
            res.append(impute_value)
        else:
            res.append(row)

    return pd.Series(res)


def impute_df(df, dic_path, inplace):
    """ cleans the df from missing/mistyped values

    Parameters
    ----------
    df : pd.DataFrame

    dic_path : str
        path containing name of the columns to clean and the upper/lower
        limits to consider point as outlier and optionnal third value is the
        imputing value

    inline : bool
        if True, performs the transformation inline
    Returns
    -------
    pd.DataFrame

    """
    if not inplace:
        # make a copy of df
        df = df[::]
    dic = {}

    with open(dic_path, 'r') as f:
        for line in f:
            # print(line)
            key_values = line.split()
            try:
                dic[key_values[0]] = (float(key_values[1]),
                                      float(key_values[2]),
                                      float(key_values[3]))
            except IndexError:
                dic[key_values[0]] = (float(key_values[1]),
                                      float(key_values[2]))
    print(dic)
    for col in dic:
        series = df[col]
        try:
            series_clean = impute_col(series,
                                      dic[col][0],
                                      dic[col][1],
                                      dic[col][2])
        except IndexError:
            print("Default value not passed - using mean")
            series_clean = impute_col(series,
                                      dic[col][0],
                                      dic[col][1],
                                      None)

        df[col] = series_clean

    return df




