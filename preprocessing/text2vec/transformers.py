"""
object classes for sklearn pipeline compatibility


"""
from .tools import avg_corpus
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator


class Text2Vector(BaseEstimator):
    """ implementation of Doc2Vec model adapted to sklearn for
    hyperparameters tuning

    """
    def __init__(self, n_components=128, dm=1, window=3):
        self.n_components = n_components
        self.dm = dm
        self.window = window

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
        self.d2v_model_ = Doc2Vec(tagged_docs, size=self.n_components,
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
        return self.d2v_model_.infer_vector(reports)


class AverageWords2Vector(BaseEstimator):
    """ trains a unsupervised word2vec model, and then transform
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
                 n_components=128, **kwargs):
        self.n_components = n_components

    def fit(self, parsed_reports, y=None, **kwargs):
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
        self.w2v_model_ = Word2Vec(parsed_reports,
                                   size=self.n_components,
                                   **kwargs)
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

    # def fit_transform(self, parsed_reports, y=None):
    #     """
    #
    #     Parameters
    #     ----------
    #     parsed_reports
    #     y
    #
    #     Returns
    #     -------
    #     """
    #     return avg_corpus(self.fit(self, parsed_reports).w2v_model_,
    #                       parsed_reports)
