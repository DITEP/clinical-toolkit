"""
unit test for OutlierRemover

@TODO add test for inplace parameter
"""
import pandas as pd
import numpy as np
import os

from numpy.testing import assert_array_equal
from preprocessing.utils import outliers

curpath = os.path.dirname(__file__)
# print('CURPATH IS: \n', curpath)

values = {'col1': np.arange(-2, 2, step=0.5),
          'col2': np.arange(0, 16, step=2),
          'col3': np.arange(1, 9, step=1)}

df = pd.DataFrame(values)

# col1  col2  col3
# 0  -2.0     0     1
# 1  -1.5     2     2
# 2  -1.0     4     3
# 3  -0.5     6     4
# 4   0.0     8     5
# 5   0.5    10     6
# 6   1.0    12     7
# 7   1.5    14     8

mean1, mean2 = np.mean(df['col1']), np.mean(df['col2'])


class TestOutliers(object):
    def setUp(self):
        print(curpath)
        return outliers.OutlierRemover(curpath + '/bounds_test.txt', False)

    def test_remove(self):
        remover = self.setUp()

        cleansed_df = remover.transform(df)

        true_col1 = np.array([mean1, mean1, mean1, mean1, 0, 0.5, 1, mean1])
        true_col2 = np.array([mean2, mean2, mean2, 6, 8, 10, 12, 14])
        true_col3 = np.array([6, 6, 6, 6, 5, 6, 7, 8])

        assert_array_equal(true_col1, cleansed_df['col1'].values)
        assert_array_equal(true_col2, cleansed_df['col2'].values)
        assert_array_equal(true_col3, cleansed_df['col3'].values)
