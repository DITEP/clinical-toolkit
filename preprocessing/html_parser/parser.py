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


class ReportsParser(BaseEstimator):
    """ a parser for html pages

    Parameters
    ----------
    strategy : string, (default='strings')
        defines the type of object returned by the transformation,
        if 'strings', each line of the returned df is string
        if 'tokens', the string is split into a list of words

    remove_sections : list, default=[]
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
        self.remove_sections = remove_sections
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
                dico = main_parser(html, i)
                merged_report = reduce_dic(dico, self.remove_sections)

                res.append(text_normalize(merged_report, self.stop_words,
                                          stem=False))

        ser_res = pd.Series(res, index=X.index)

        if self.strategy == 'strings':
            # merge tokens into a string
            return ser_res.apply(lambda x: ' '.join(x))
        elif self.strategy == 'tokens':
            return ser_res
        else:
            return ValueError("Expected 'tokens' or 'string' in strategy")
