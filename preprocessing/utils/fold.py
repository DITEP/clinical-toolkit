"""
As data may come from different sources, it is best to retrieve all the bases 
into one single dataframe that would enables fetching the features very 
easily, as well as the dates at which the events/measures occured.

Doing so allows to retrieve the full timelines of the patients and 
therefore complete various tasks.  

The objective of this module is to parse the databases available in order to
have each one of them organized as

key1 | key2 | feature_name | value | date

"""
import pandas as pd

from sklearn.base import BaseEstimator


class Folder(BaseEstimator):
    """  This object enables "unfolding" the features of a DataFrame, 
    which means for a df that has 5 feature columns for instance, 
    the unfolding would result in two feature columns: one is for the feature 
    name and the other is the feature value.

    All the attributes are column names to indicate how to unfold the dataframe

    Parameters
    ----------
    key1 : str
        indicator of the primary key indicator

    key2 : str, (optionnal?)
        seconday key

    features : list
        column names that contain the feature

    date : str
        name of the date column,


    """
    def __init__(self, key1, key2, features, date):
        self.key1 = key1
        self.key2 = key2
        self.features = features
        self.date = date

    def fit(self, X, y=None):
        return self

    def transform(self, df_base, y=None):
        """ 

        Parameters
        ----------
        df_base : pandas DataFrame
        
        y : None

        Returns
        -------
        pandas.DataFrame
        columns are [key1, key2, feature, value, date] where feature contains
        the features names and values are the values.

        """
        df_res = pd.DataFrame(None, columns=[self.key1, self.key2, 'feature',
                                             'value', 'date'])
        if len(self.features) > 1:
            for _, row in df_base.iterrows():
                dico = {self.key1: [], self.key2: [], 'feature': [],
                        'value': [], 'date': []}
                for cur_feat in self.features:
                    dico[self.key1] += [row[self.key1]]
                    dico[self.key2] += [row[self.key2]]

                    dico['feature'] += [cur_feat]
                    dico['value'] += [row[cur_feat]]
                    dico['date'] += [row[self.date]]
                df_cur = pd.DataFrame(dico)
                df_res = df_res.append(df_cur, ignore_index=True)

        # only one feature
        else:
            # df_res['feature'] = df_base[self.features]
            for _, row in df_base.iterrows():
                dico = {self.key1: row[self.key1],
                        self.key2: row[self.key2],
                        'feature': self.features[0],
                        'value': row[self.features[0]],
                        'date': row[self.date]}

                df_res = df_res.append(dico, ignore_index=True)

        return df_res
