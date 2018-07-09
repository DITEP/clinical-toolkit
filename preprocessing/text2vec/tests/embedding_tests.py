"""
script to test reptk.features.embedding
"""
import numpy as np

from numpy.testing import assert_array_almost_equal
from preprocessing.text2vec import tools
from gensim.models import word2vec


corpus_test = ['la programmation en python c est genial'.split(' '),
              'c est un langage souple qui contient de nombreuses libraires'.split(' '),
              'l omelette au fromage est meilleure qu au sucre'.split(' ')]

#setting random seed for reproducibility
w2v = word2vec.Word2Vec(corpus_test, size=3, seed=0, min_count=1,
                        workers=1).wv


class TestEmbeddings(object):
    def setUp(self):
        pass

    def test_avg_doc(self):
        doc1 = 'omelette au champignon'.split(' ')
        doc2 = 'la programmation en java c est pas cool'.split(' ')
        doc3 = 'nose pour les tests unitaires c est genial'.split(' ')
        doc4 = 'rien a voir'.split(' ')

        avg1 = tools.avg_document(w2v, doc1)
        avg2 = tools.avg_document(w2v, doc2)
        avg3 = tools.avg_document(w2v, doc3)
        avg4 = tools.avg_document(w2v, doc4)

        expected1 = np.mean([w2v['omelette'],
                            w2v['au']], axis=0)
        expected2 = np.mean([w2v['la'],
                             w2v['programmation'],
                             w2v['en'],
                             w2v['c'],
                             w2v['est']], axis=0)
        expected3 = np.mean([w2v['c'],
                             w2v['est'],
                             w2v['genial']], axis=0)
        expected4 = np.zeros(3)

        assert_array_almost_equal(expected1, avg1)
        assert_array_almost_equal(expected2, avg2)
        assert_array_almost_equal(expected3, avg3)
        assert_array_almost_equal(expected4, avg4)
