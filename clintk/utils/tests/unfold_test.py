"""
test scipt for unfold
"""
import pandas as pd
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_list_equal
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.utils.unfold import transform_and_label, Unfolder
from preprocessing.text_parser.parser import ReportsParser
from sklearn.pipeline import Pipeline


dico_input = {'key1': ['1', '1', '1', '2', '2', '2', '2', '2', '2'],
              'key2': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'a2',
                       'a2', 'a2'],
              'feature': ['feat1', 'feat2', 'feat3', 'feat1',
                          'feat2', 'feat3', 'feat1', 'feat2',
                          'feat3'],
              'value': [3, 'Text1', 5, 1, 'teXt2', 3, 3, 'tExt3', 5],
              'date': ['2018-06-14', '2018-06-14', '2018-06-14',
                       '2018-05-22', '2018-05-22', '2018-05-22',
                       '2017-03-01', '2017-03-01', '2017-03-01']}

df_input = pd.DataFrame(dico_input, dtype=object)

""" >>> df_input
    key1 key2 feature value        date
0    1   a1   feat1      3  2018-06-14
1    1   a1   feat2  Text1  2018-06-14
2    1   a1   feat3      5  2018-06-14
3    2   a2   feat1      1  2018-05-22
4    2   a2   feat2  teXt2  2018-05-22
5    2   a2   feat3      3  2018-05-22
6    2   a2   feat1      3  2017-03-01
7    2   a2   feat2  tExt3  2017-03-01
8    2   a2   feat3      5  2017-03-01
"""


class TestUnfold(object):
    def setUp(self):
        return df_input

    def test_tfidf_label(self):
        """
        Tests transform_and_label on tfidf transformer
        """
        df = self.setUp()
        df = df[df['feature'] == 'feat2']
        df.index = pd.RangeIndex(len(df.index))

        df_res = transform_and_label(df, 'key1', 'key2', 'date', 'feature',
                                     'value', TfidfVectorizer)

        expected = {'key1': ['1',  '2',  '2'],
                    'key2': ['a1', 'a2','a2'],
                    'feature': ['feat2_0', 'feat2_1', 'feat2_2'],
                    'value': [1, 1, 1],
                    'date': ['2018-06-14', '2018-05-22', '2017-03-01']}

        df_expected = pd.DataFrame(expected)

        assert_array_equal(df_expected, df_res.values)
        assert_list_equal(list(df_expected.columns), list(df_res.columns))


    def test_pipeline(self):
        """
        Tests transform_and_label on custom ReportParser transformer
        """
        df = self.setUp()
        df = df[df['feature'] == 'feat2']
        df.index = pd.RangeIndex(len(df.index))

        pipeline = Pipeline

        df_res = transform_and_label(df, 'key1', 'key2', 'date', 'feature',
                                     'value', pipeline,
                                     steps=[('parser',
                                             ReportsParser(headers=None)),
                                            ('tfidf', TfidfVectorizer())])

        expected = {'key1': ['1', '2', '2'],
                    'key2': ['a1', 'a2', 'a2'],
                    'feature': ['feat2_0', 'feat2_1', 'feat2_2'],
                    'value': [1, 1, 1],
                    'date': ['2018-06-14', '2018-05-22', '2017-03-01']}

        df_expected = pd.DataFrame(expected)

        assert_array_equal(df_expected, df_res.values)
        assert_list_equal(list(df_expected.columns), list(df_res.columns))


    def test_unfold_numeric(self):
        """
        Tests unfold on small df
        """
        df = self.setUp()
        df = df[df['feature'] != 'feat2']

        unfolder = Unfolder('key1', 'key2', 'feature', 'value', 'date')
        unfolder.fit(df)

        unfolded_df = unfolder.unfold()

        # unfolded_df = unfold(df, 'key1', 'key2', 'feature', 'value', 'date')

        expected = {'key1': ['1', '2', '2'],
                    'key2': ['a1', 'a2', 'a2'],
                    'date': ['2018-06-14', '2018-05-22', '2017-03-01'],
                    'feat1': [3, 1, 3],
                    'feat3': [5, 3, 5]}

        df_expected = pd.DataFrame(expected)

        assert_array_equal(df_expected, unfolded_df.values)
        assert_list_equal(list(df_expected.columns), list(unfolded_df.columns))
