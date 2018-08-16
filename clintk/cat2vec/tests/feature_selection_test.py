"""


"""
import pandas as pd
import numpy as np

from preprocessing.cat2vec.feature_selection import LassoSelector
from numpy.testing import assert_array_equal


values = {'feature1': [0, 0, 1, 1, 0],
          'feature2': [0, 1, 1, 0, 1],
          'feature3': [1, 0, 0, 0, 0],
          'feature4': [1, 0, 0, 0, 1]}

coefficients = {'coef': [0, 4.5, -1.2, 0.5],
                'feature_name': ['feature1', 'feature2', 'feature3',
                                 'feature4']}


df = pd.DataFrame(values)
#         feature1  feature2  feature3  feature4
# 0         0         0         1         1
# 1         0         1         0         0
# 2         1         1         0         0
# 3         1         0         0         0
# 4         0         1         0         1

df_coef = pd.DataFrame(coefficients)
#     coef    feature_name
# 0   0.0     feature1
# 1   4.5     feature2
# 2  -1.2     feature3
# 3   0.5     feature4


class TestTransformation(object):
    def SetUp(self):
        return self

    def test_fit_transform(self):
        selector = LassoSelector(n_features=2,
                                 lasso_coefs=df_coef,
                                 feature_col='feature_name',
                                 coef_col='coef')
        # selector.fit(df_coef.feature_name, df_coef.coef)

        x_res = selector.transform(df)

        x_expected = np.array([[0, 1], [1, 0], [1, 0], [0, 0], [1, 0]])

        assert_array_equal(x_expected, x_res)






