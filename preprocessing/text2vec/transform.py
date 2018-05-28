"""
This script is designed to extract features from text reports.
The first main step to do so is the tokenization, which
is made in `tokenize_report` that takes as input a pandas
DataFrame and returns a series of the tokenized reports.


@warning whole script is deprecated
"""
from ..html_parser.text_parser import main_parser
from ..html_parser.section_manager import reduce_dic

from sklearn.feature_extraction import text

from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize

import pandas as pd

# from gensim.models import Word2Vec
# from gensim import corpora
# import numpy as np


def tokenize_report(df, remove_section=[], col_name='report'):
    """ Process the html report

    .. note:: Deprecated
        use `transformers.HTMLNormalize`  instead for consistency
        with sklearn

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        dataframe containing information about patients, usually they
        are id, report_date, report
    remove_section : list
        list containing the names of the sections to be removes from
        the report
    col_name : string
        name of the columns containing the text reports in html format

    Returns
    -------
    pandas.core.series.Series
        each report parsed, normlized and tokenized

    Examples
    --------
    >>> token_series = tokenize_report(df,[])

    #to get a series of strings instead of tokens:
    >>> token_series.apply(lambda tokens: ' '.join(tokens))

    """
    res = []
    for i, row in df.iterrows():
        html = row[col_name]
        dico = main_parser(html, i)
        merged_report = reduce_dic(dico, remove_section)

        res.append(text_normalize(merged_report, stem=False))

    return pd.Series(res, index= df.index)


def vectorize_df(df, colname='text_tokens', ):
    """
    colname indicates the columns where text reports are
    already normalized and tokenized

    :param df:
    :return:
    """
    X = df[colname].values
    # res = np.zeros(texts.shape)
    # freq_dic = corpora.Dictionary(texts)
    counter = text.CountVectorizer()
    vectorizer = text.TfidfTransformer()
    counts = counter.fit_transform(X)
    features = vectorizer.fit_transform(counts)

    df['text_features'] = features

    return df


def text_normalize(text, stop_words, stem=False):
    """ This functions performs the preprocessing steps
    needed to optimize the vectorization, such as normalization
    stop words removal, lemmatization etc...

    stemming for french not accurate enough yet
    @TODO lemmatization for french

    Parameters
    ----------
    text: string
        text to normalize

    Returns
    -------
    string
        same text as input but cleansed and normalized
    """
    sw = stopwords.words('french') + stop_words
    tokens = word_tokenize(text, 'french')

    tokens_filter = [word for word in tokens if word not in sw]

    if stem:
        stemmer = FrenchStemmer()
        tokens_filter = [stemmer.stem(word) for word in tokens_filter]

    return tokens_filter   # " ".join(tokens_filter)  #


def get_X(df, stop_section=['examen du patient']):
    """ get X matrice of text feature and removes useless
    sections
    @deprecated
    :param df:
    :param stop_section:
    :return:
    """
    mask = ~df['section'].isin(stop_section)

    X = df['text'][mask]

    return X, mask
