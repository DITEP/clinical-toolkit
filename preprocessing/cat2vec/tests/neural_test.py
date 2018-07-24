"""
Unit test suite for NeuralVectorizer

"""
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from numpy.testing import assert_array_equal, assert_array_almost_equal
from preprocessing.cat2vec import neural_embedding, tools

# global variables definition

dic = {'ids': [0, 0, 1, 2, 2, 2, 0],
       'values': ['first cat', 'first cat', 'second cat', 'first cat',
                  'second cat', 'third cat', 'third cat']}

df = pd.DataFrame(dic, index=None)
df['values'] = tools.normalize_cat(df['values'])

corpus = [['first', 'cat', 'first', 'cat', 'third', 'cat'],
          ['second', 'cat'],
          ['first', 'cat', 'second', 'cat', 'third', 'cat']]

print(df)
size, min_count, window, sg = 3, 1, 1, 1


w2v_true = Word2Vec(corpus, size=size, window=window, sg=sg,
                    min_count=min_count, seed=0)


class TestVectorize(object):
    def setUp(self):
        return neural_embedding.W2VVectorizer('ids', 'values', size,
                                              min_count, sg, window)

    def test_fitter(self):
        embedder = self.setUp()

        embedder.fit(df)
        w2v_pred = embedder.w2v_

        assert_array_equal(w2v_true.wv.vocab.keys(), w2v_pred.wv.vocab.keys())
        assert_array_equal(w2v_true.corpus_count, w2v_pred.syn0.shape[1])
        assert_array_equal(w2v_true.wv.vectors, w2v_pred.wv.vectors)

    def test_transformer(self):
        embedder = self.setUp().fit(df)

        v1 = (w2v_true['first'] + w2v_true['cat']) / 2
        v2 = (w2v_true['first'] + w2v_true['cat']) / 2
        v3 = (w2v_true['second'] + w2v_true['cat']) / 2
        v4 = (w2v_true['first'] + w2v_true['cat']) / 2
        v5 = (w2v_true['second'] + w2v_true['cat']) / 2
        v6 = (w2v_true['third'] + w2v_true['cat']) / 2
        v7 = (w2v_true['third'] + w2v_true['cat']) / 2
        true_col = np.array([v1, v2, v3, v4, v5, v6, v7])

        col_embed = embedder.transform(df)['values_embedded']

        for i, j in enumerate(true_col):
            assert_array_almost_equal(j, col_embed[i])

    def test_pretrained(self):
        embedder = self.setUp()
        path = 'data/wiki.fr.vec'

        embedder.fit_pretrained(path, limit=10)



