"""
object classes for sklearn pipeline comprehension


@TODO remplacer word2vec par modèle doc2vec de gensim (
@TODOhttp://yaronvazana.com/2018/01/20/training-doc2vec-model-with-gensim/)

"""
from .embedding import avg_corpus
from gensim.models import word2vec
from sklearn.base import BaseEstimator

import pandas as pd


class Text2Vector(BaseEstimator):
    """ trains a unsupervised word2vec model, and then transform
    text data according to it
    This function is only for convenience in using word2vec in a pipeline

    Parameters
    ----------
    n_components : int, default=128
        dimension of the embedding vector

    sg : int, {0,1}, default=1
        Defines training algorithm, 1 for skip-gram, otherwise CBOW

    window : int, default=5
        The maximum distance between the current an predicted word in a
        sentence

    alpha : float, default=0.025
        The initial learning rate

    min_alpha : float, default=1e4
        Learning rate will linearly decrease to min_alpha during training

    seed : int, default=0
        random seed

    min_count : int, default=5
        ignore words with total frequency lower than this

    max_vocab_size : int, default=None
        Limits the RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10
        million word types need about 1GB of RAM.
        Set to None for no limit

    sample : float, default=1e-3
         The threshold for configuring which higher-frequency words
          are randomly downsampled, useful range is (0, 1e-5)

    workers : int, default=2
        number of worker threads (-1 for all)

    hs : int, {0,1}, default=0
        If 1, hierarchical softmax will be used for model training.
        if 0, and negative > 0, negative sampling

    negative : int, default=5
         if > 0 , negative sample used
         if = 0, no negative sampling used

    cbow_mean : int, {0,1}, default=1
        If 0, use the sum of the context word vectors.
        If 1, use the mean, only applies when cbow is used

    epochs : int, default=5
        Number of epochs over the corpus

    sorted_vocab : int, {0,1}, default=1
        If 1, sort the vocab by descending frequency before assiging
        word indexes
    """
    def __init__(self,
                 n_components=128, **kwargs):
        self.w2v = lambda corpus:  \
            word2vec.Word2Vec(corpus,
                              size=n_components,
                              **kwargs)
                              # sg=sg, window=window,
                              # alpha=alpha, min_alpha=min_alpha,
                              # seed=seed, min_count=min_count,
                              # max_vocab_size=max_vocab_size,
                              # sample=sample, workers=workers,
                              # hs=hs, negative=negative,
                              # cbow_mean=cbow_mean, iter=epochs,
                              # sorted_vocab=sorted_vocab)
        # trained w2v model
        self.w2v_model_ = None

    def fit(self, parsed_reports, y=None):
        """ Trains the word2vec model with given corpus
        as input

        Parameters
        ----------
        parsed_reports : iterable of ierables
            contains parsed tokenized reports

        y : None

        Returns
        -------
        """
        self.w2v_model_ = self.w2v(parsed_reports)
        return self

    def transform(self, parsed_reports):
        """ Turns the documents into vector by averaging
        over all the words

        Parameters
        ----------
        parsed_reports : iterable of iterables

        Returns
        -------

        """
        return avg_corpus(self.w2v_model_, parsed_reports)

    def fit_transform(self, parsed_reports, y=None):
        """

        Parameters
        ----------
        parsed_reports
        y

        Returns
        -------
        """
        return avg_corpus(self.w2v(parsed_reports), parsed_reports)




