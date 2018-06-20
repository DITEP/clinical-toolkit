"""
testing Folder class
"""
import pandas as pd
import numpy as np

from preprocessing.utils.fold import Folder
from numpy.testing import assert_array_equal


row1 = [1, 'a1', 3, 4, 5, '2018-06-14']
row2 = [2, 'a2', 1, 2, 3, '2018-05-22']
row3 = [2, 'a2', 3, 4, 5, '2017-03-01']

values = np.array([np.array(row1),
                  np.array(row2),
                  np.array(row3)])

df = pd.DataFrame(values, columns=['key1', 'key2', 'feat1', 'feat2', 'feat3',
                                   'event_date'])

"""
>>> df

    key1 key2 feat1 feat2 feat3  event_date
0    1   a1     3     4     5   2018-06-14
1    2   a2     1     2     3   2018-05-22
2    2   a2     3     4     5   2017-03-01
"""

class TestFolder(object):
    def setUp(self):

        return df

    def test_multiple_fold(self):
        df_base = self.setUp()

        unfold = Folder('key1', 'key2', ['feat1', 'feat2', 'feat3'],
                          'event_date')

        df_res = unfold.transform(df_base)

        dico_expected = {'key1': ['1', '1', '1', '2', '2', '2', '2', '2', '2'],
                         'key2': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'a2',
                                  'a2', 'a2'],
                         'feature': ['feat1', 'feat2', 'feat3', 'feat1',
                                     'feat2', 'feat3', 'feat1', 'feat2',
                                     'feat3'],
                         'value': ['3', '4', '5', '1', '2', '3', '3', '4', '5'],
                         'date': ['2018-06-14', '2018-06-14', '2018-06-14',
                                  '2018-05-22', '2018-05-22', '2018-05-22',
                                  '2017-03-01', '2017-03-01', '2017-03-01']}
        df_expected = pd.DataFrame(dico_expected, dtype=object)

        assert_array_equal(df_expected.values, df_res.values)


    def test_unique_fold(self):
        df_base = self.setUp().drop(['feat2', 'feat3'], axis=1)

        unfold = Folder('key1', 'key2', ['feat1'], 'event_date')

        df_res = unfold.transform(df_base)

        df_expected = pd.DataFrame({'key1': ['1', '2', '2'],
                                    'key2': ['a1', 'a2', 'a2'],
                                    'feature': ['feat1', 'feat1', 'feat1'],
                                    'value': ['3', '1', '3'],
                                    'date': ['2018-06-14', '2018-05-22',
                                             '2017-03-01']})

        assert_array_equal(df_expected, df_res)
