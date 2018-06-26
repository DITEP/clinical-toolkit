"""
object to parse reports written in html

compatible with scikit-learn transformer API

"""
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator
from .section_manager import reduce_dic
from .text_parser import main_parser, clean_string
from preprocessing.text2vec.tools import text_normalize
from multiprocessing.pool import Pool


class ReportsParser(BaseEstimator):
    """ a parser for html pages

    Parameters
    ----------
    strategy : string, (default='strings')
        defines the type of object returned by the transformation,
        if 'strings', each line of the returned df is string. 'strings' is to
        be used for CountVectorizer and TFiDFVectorizer
        if 'tokens', the string is split into a list of words. 'tokens' is to
        be used for gensim's Word2Vec and Doc2Vec models

    remove_sections : list, default=[]
        list containing the names of the sections to be removes from

    remove_tags : list, default=['h4', 'table', 'link', 'style']
        list of tags to remove from html  page

    headers : string, default='h3
        name of the html tag that delimits the sections in the page

    stop_words : list, default=[]
        additional words to remove from the text, specific to the kind
        of parsed document

    verbose : bool, default=Fale

    """
    def __init__(self,
                 strategy='strings',
                 remove_sections=[],
                 remove_tags=['h4', 'table', 'link', 'style'],
                 col_name='report',
                 headers='h3',
                 stop_words=[],
                 verbose=False,
                 n_jobs=1):

        self.strategy = strategy
        self.remove_sections = remove_sections
        self.tags = remove_tags
        self.headers = headers
        self.colName = col_name
        self.verbose = verbose
        self.stop_words = stop_words
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
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
        # res = []
        # for html in X:
        #     res.append(self.fetch_doc(html))
        if self.n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(self.n_jobs )

        res = pool.map(self.fetch_doc, X)

        pool.close()
        pool.join()


        ser_res = pd.Series(res) #, index=X.index)

        return ser_res

    def fetch_doc(self, html):
        if self.headers is None:
            # html is not structured
            text = clean_string(BeautifulSoup(str(html),
                                              'html.parser').text)
            text = text_normalize(text, self.stop_words, stem=False)

        # parse html split into self.headers
        else:
            dico = main_parser(html, self.verbose)
            text = reduce_dic(dico, self.remove_sections)

            text = text_normalize(text, self.stop_words,
                                  stem=False)
        if self.strategy == 'strings':
            return ' '.join(text)
        else:
            return text
