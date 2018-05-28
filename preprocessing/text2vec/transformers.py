"""
object classes for sklearn pipeline comprehension



"""
from .transform import text_normalize
from .embedding import avg_corpus
from ..html_parser.section_manager import reduce_dic
from ..html_parser.text_parser import main_parser, clean_string

from gensim.models import word2vec
from bs4 import BeautifulSoup

import pandas as pd


class HTMLParser(object):
    """ a parser for html pages

    Parameters
    ----------
    strategy : string, (default='strings')
        defines the type of object returned by the transformation,
        if 'strings', each line of the returned df is string
        if 'tokens', the string is split into a list of words

    remove_sections : list
        list containing the names of the sections to be removes from

    remove_tags : list
        list of tags to remove from html  page

    headers : string
        name of the html tag that delimits the sections in the page

    stop_words : list, default=[]
        additional words to remove from the text, specific to the kind
        of parsed document

    verbose : bool

    """
    def __init__(self,
                 strategy='strings',
                 remove_sections=[],
                 remove_tags=['h4', 'table', 'link', 'style'],
                 col_name='report',
                 headers='h3',
                 stop_words=[],
                 verbose=False):

        self.strategy = strategy
        self.sections = remove_sections
        self.tags = remove_tags
        self.headers = headers
        self.colName = col_name
        self.verbose = verbose
        self.stop_words = stop_words

    def fit(self, X, y):
        return self

    def transform(self, X):
        """

        Parameters
        ----------
        X : pd.Series or DataFrame

        Returns
        -------
        pd.Series
            each entry is either a string or list of words depending on
            the strategy
        """
        if type(X) == pd.DataFrame:
            # then turn it into a Series
            X = X[self.colName]
        res = []
        for i, html in X.iteritems():
            if self.headers is None:
                # html is not structured
                text = clean_string(BeautifulSoup(html, 'html.parser').text)
                res.append(text_normalize(text, self.stop_words, stem=False))
            else:
                # dico = main_parser(clean_string(html), i) TODO <--
                dico = main_parser(html, i)
                merged_report = reduce_dic(dico, self.sections)

                res.append(text_normalize(merged_report, self.stop_words, stem=False))

        ser_res = pd.Series(res, index=X.index)

        if self.strategy == 'strings':
            # merge tokens into a string
            return ser_res.apply(lambda x: ' '.join(x))
        elif self.strategy == 'tokens':
            return ser_res
        else:
            return ValueError("Expected 'tokens' or 'string' in strategy")


class Text2Vector(object):
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
                 n_components=128, sg=1, window=5, alpha=0.025,
                 min_alpha=1e-4, seed=0, min_count=5,
                 max_vocab_size=None, sample=1e-3, workers=2,
                 hs=0, negative=5, cbow_mean=1, epochs=5,
                 sorted_vocab=1):
        self.w2v = lambda corpus:  \
            word2vec.Word2Vec(corpus,
                              size=n_components,
                              sg=sg, window=window,
                              alpha=alpha, min_alpha=min_alpha,
                              seed=seed, min_count=min_count,
                              max_vocab_size=max_vocab_size,
                              sample=sample, workers=workers,
                              hs=hs, negative=negative,
                              cbow_mean=cbow_mean, iter=epochs,
                              sorted_vocab=sorted_vocab)
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




