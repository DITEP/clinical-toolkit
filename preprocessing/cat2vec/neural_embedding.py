"""
Embedding of high cardinality categorical variables using Wod2Vec

Embeding is done in two parts: first we group the categories by patient so
that each one contains a corpus of them, allowing to process the
concatenation of categories as text
"""
import pandas as pd
from gensim.models import Word2Vec

from ..text2vec import tools


class NeuralVectorizer(object):
    """
    Parameters
    ----------
    group_key : str
        name of the column to group

    category_col : str
        name of the column containing the categorical variables

    size : int, default=128
        dimension of the embedding vector

    min_count : int, default=1
        minimum amount of instances to integrate it to the model

    sg : int {0, 1}, default=1
        0 for skip-gram word2vec model
        1 for CBOW (best suited for small datasets)

    window : int, default=3
        size of the context

    strategy : str {'tokens', 'strings'}, default='tokens'
        if 'tokens', categories containing several words are split
        else, each category is considered as a word

            """
    def __init__(self, group_key, category_col,
                 size=128, min_count=1, sg=1,
                 window=3, strategy='tokens',
                 seed=0):
        self.key = group_key,
        self.cat_col = category_col
        self.size = size
        self.min_count = min_count
        self.sg = sg
        self.window = window
        self.strategy = strategy
        self.seed = seed
        self.w2v_ = None

    def fit(self, X, y=None):
        """ fits the model by grouping categories by group_key in order to
        embed categories as text

        Parameters
        ----------
        X : pd.DataFrame
        y

        Returns
        -------

        """
        df_grouped = X.groupby(self.key).agg({self.cat_col: ' '.join})
        df_grouped[self.cat_col] = df_grouped[self.cat_col] \
            .apply(lambda s: s.split(' '))

        self.w2v_ = Word2Vec(df_grouped[self.cat_col],
                             size=self.size,
                             window=self.window,
                             min_count=self.min_count,
                             sg=self.sg,
                             seed=self.seed)
        return self

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : pd.DataFrame

        y : None

        Returns
        -------

        """
        categories = X[self.cat_col].apply(lambda s: s.split(' '))

        vectors = categories.apply(lambda cat: tools.avg_document(self.w2v_, cat))

        X[self.cat_col + '_embedded'] = vectors

        return X
