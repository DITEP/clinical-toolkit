"""
object classes for sklearn pipeline compatibility


"""
import numpy as np

from .tools import avg_corpus
from gensim.models import KeyedVectors, Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.base import BaseEstimator


class Text2Vector(BaseEstimator):
    """ implementation of Doc2Vec model adapted to sklearn for
    hyperparameters tuning

    """
    def __init__(self, n_components=128, dm=1, window=3):
        self.n_components = n_components
        self.dm = dm
        self.window = window
        self.d2v_model_ = None

    def fit(self, reports, y=None, **kwargs):
        """ tags reports (for gensim's model consistence) and trains Doc2Vec
        model on the corpus

        Parameters
        ----------
        reports : iterable of iterables
            list of tokenized reports

        y : not used, default=None

        Returns
        -------

        """
        tagged_docs = [TaggedDocument(j, 'doc_{}'.format(i))
                       for i, j in enumerate(reports)]

        # self.d2v_model_ = self.d2v(tagged_docs)
        self.d2v_model_ = Doc2Vec(tagged_docs, vector_size=self.n_components,
                                  dm=self.dm, window=self.window,
                                  **kwargs)

        return self

    def transform(self, reports):
        """ transforms reports in embedding space based on previously trained
        Doc2Vec model

        Parameters
        ----------
        reports : iterable of iterables
            list of tokenized reports

        Returns
        -------
        np.ndarray
            vectorized reports
        """
        return np.array([self.d2v_model_.infer_vector(document) for document
                         in reports])


class AverageWords2Vector(BaseEstimator):
    """ trains a unsupervised word2vec model, and then fold
    text data according to it
    This function is only for convenience in using word2vec in a pipeline

    Parameters
    ----------
    n_components : int, default=128
        dimension of the embedding vector

    kwargs
    additionnal arguments to pass to gensim.Word2Vec (see appropriate
    documentation for details)
    """
    def __init__(self,
                 n_components=128):
        self.n_components = n_components
        self.w2v_ = None

    def fit(self, parsed_reports, y=None, **kwargs):
        """ Trains the word2vec model with given corpus
        as input

        Parameters
        ----------
        parsed_reports : iterable of iterables
            contains parsed tokenized reports

        y : None

        Returns
        -------
        """
        self.w2v_ = Word2Vec(parsed_reports, size=self.n_components,
                             **kwargs).wv
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
        return avg_corpus(self.w2v_, parsed_reports)

    def fit_pretrained(self, path, **kwargs):
        """ fits a pretrained model from
        https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

        Parameters
        ----------
        path : str
            path to the model

        Returns
        -------

        """
        self.w2v_ = KeyedVectors.load_word2vec_format(path, **kwargs)

        return self
