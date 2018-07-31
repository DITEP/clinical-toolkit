"""
object to parse text reports, compatible with scikit-learn transformer API

The format of typical reports to be parsed can be found in data/ directory of
this repo. `ReportsParser` enables choosing custom :

* section delimiters with `headers` attribute
* tags that dont contain informative texte (style tag for instance) with
  `remove_tags`
* additional stop words, that may be specific to a corpus or a task

"""
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator
from .section_manager import reduce_dic
from .text_parser import main_parser, clean_string
from preprocessing.text2vec.tools import text_normalize
from multiprocessing.pool import Pool


class ReportsParser(BaseEstimator):
    """ a parser for html-like text reports

    Parameters
    ----------
    strategy : string, default='strings'
        defines the type of object returned by the transformation,
        if 'strings', each line of the returned df is string. 'strings' is to
        be used for CountVectorizer and TFiDFVectorizer
        if 'tokens', the string is split into a list of words. 'tokens' is to
        be used for gensim's Word2Vec and Doc2Vec models

    remove_sections : list, default=[]
        list containing names of the sections to remove

    remove_tags : list, default=['h4', 'table', 'link', 'style']
        list of tags to remove from html  page

    headers : str or None, default='h3
        name of the html tag that delimits the sections in the

    is_html : bool, default=True
        boolean indicating weather the structure of the reports is strictly html
        format or not.
        Check documentation usage for details

    stop_words : list, default=[]
        additional words to remove from the text, specific to the kind
        of parsed document

    verbose : bool, default=False

    n_jobs : int, default=1
        number of CPU cores to use, if -1 then all the available one are used

    See Also
    --------
    .text_parser module : which contains the core functions to parse each text

    """
    def __init__(self,
                 strategy='strings',
                 remove_sections=[],
                 remove_tags=['h4', 'table', 'link', 'style'],
                 col_name='report',
                 headers='h3',
                 is_html=True,
                 stop_words=[],
                 verbose=False,
                 n_jobs=1):

        self.strategy = strategy
        self.remove_sections = remove_sections
        self.tags = remove_tags
        self.headers = headers
        self.is_html = is_html
        self.col_name = col_name
        self.verbose = verbose
        self.stop_words = stop_words
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ parses the reports in input

        Parameters
        ----------
        X : pd.Series or DataFrame
            each entry is a string defining a report

        Returns
        -------
        pd.Series
            each entry is either a string or list of words depending on
            the strategy
        """
        if type(X) == pd.DataFrame:
            # then turn it into a Series
            X = X[self.col_name]
        if self.n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(self.n_jobs)

        res = pool.map(self._fetch_doc, X)

        pool.close()
        pool.join()

        ser_res = pd.Series(res)

        return ser_res

    def _fetch_doc(self, html):
        """ parses one html document using `self` parameters

        Method is protected as it is only made to be used to facilitate the
        serialization of the main loop in `transform`


        Parameters
        ----------
        html : str

        Returns
        -------
        str or list of str
            depending of `self.strategy`


        """
        if self.headers is None:
            # html is not structured
            text = clean_string(BeautifulSoup(str(html),
                                              'html.parser').text)
            text = text_normalize(text, self.stop_words, stem=False)

        # parse html split into self.headers
        else:
            dico = main_parser(html, self.is_html, self.verbose,
                               headers=self.headers)
            text = reduce_dic(dico, self.remove_sections)

            text = text_normalize(text, self.stop_words,
                                  stem=False)
        if self.strategy == 'strings':
            return ' '.join(text)
        else:
            return text
