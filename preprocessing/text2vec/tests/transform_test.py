"""
script to test functions in preprocessing.text2vec.transform
"""
from preprocessing.text2vec import transform
from nose.tools import assert_list_equal


class TestTransform(object):
    def setUp(self):
        pass

    def test_normalizer(self):
        s = 'alfred le chat de mon voisin était rose avec des tâches'

        res = transform.text_normalize(s, ['chat'])

        expected = ['alfred', 'voisin', 'rose', 'tâches']

        assert_list_equal(expected, res)

   







