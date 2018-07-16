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

from multiprocessing.pool import Pool


class Folder:
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
        secondary key

    features : list
        column names that contain the feature

    date : str
        name of the date column,

    n_jobs : int
        number of CPUs to use for computation. If -1, all the available cores
        are used



    """
    def __init__(self, key1, key2, features, date, n_jobs=1):
        self.key1 = key1
        self.key2 = key2
        self.features = features
        self.date = date
        self.n_jobs = n_jobs

    def fold(self, df_base):
        """ 

        Parameters
        ----------
        df_base : pandas DataFrame
        
        Returns
        -------
        pandas.DataFrame
            columns are [key1, key2, feature, value, date] where feature
            contains the features names and values are the values.

        """
        columns = [self.key1, self.key2, 'feature', 'value', 'date']
        if self.n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(self.n_jobs)

        if len(self.features) > 1:
            dicts = pool.map(self.fold_several_features, df_base.iterrows())
            pool.close()
            pool.join()

            merged_dict = {k: [] for k in columns}
            for key in dicts[0]:
                for dico in dicts:
                    merged_dict[key] += dico[key]

        # only one feature
        else:
            dicts = pool.map(self.fold_one_feature, df_base.iterrows())

            pool.close()
            pool.join()

            merged_dict = {k: [d[k] for d in dicts] for k in dicts[0]}

        return pd.DataFrame(merged_dict)


    def fold_several_features(self, row):
        # fetching value of the row, dropping index
        _, row = row
        dico = {self.key1: [], self.key2: [], 'feature': [],
                'value': [], 'date': []}
        for cur_feat in self.features:
            dico[self.key1] += [row[self.key1]]
            dico[self.key2] += [row[self.key2]]

            dico['feature'] += [cur_feat]
            dico['value'] += [row[cur_feat]]
            dico['date'] += [row[self.date]]

        return dico

    def fold_one_feature(self, row):
        _, row = row

        dico = {self.key1: row[self.key1],
                self.key2: row[self.key2],
                'feature': self.features[0],
                'value': row[self.features[0]],
                'date': row[self.date]}

        return dico
