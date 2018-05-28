"""
script to test functions in reptk.features.transform
"""
import pandas as pd

from reptk.features import transform
from nose.tools import assert_list_equal
from numpy.testing import assert_array_equal


class TestTransform(object):
    def setUp(self):
        pass

    def test_normalizer(self):
        s = 'alfred le chat de mon voisin était rose avec des tâches'

        res = transform.text_normalize(s, ['chat'])

        expected = ['alfred', 'voisin', 'rose', 'tâches']

        assert_list_equal(expected, res)

   







