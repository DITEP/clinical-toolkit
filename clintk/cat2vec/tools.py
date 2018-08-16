"""
sample script for categorical encoding
"""

import re
import unidecode
import pandas as pd

import preprocessing.text2vec.tools


def normalize_cat(X, strat='tokens'):
    """ normalize categories in a

    Parameters
    ----------
    X : iterable

    strat : str, ('tokens', 'strings'), default='tokens"
        if 'tokens', words in a category are kept split (use this for embedding
        categories by a nlp aproach)
        if 'strings', each category is considered as a single word

    Returns
    -------
    pandas.Series
        same size as input, each entry corresponding to the normalized
        category name

    """
    res = []
    for x in X:
        try:
            x = unidecode.unidecode(x)
        except AttributeError:
            x = str(x)
            x = unidecode.unidecode(x)
        x = x.lower()
        patt = re.compile('[\W_]+')
        x_norm = preprocessing.text2vec.tools.text_normalize(patt.sub(' ', x),
                                                             ['sw'])
        res.append(x_norm)
    if strat == 'tokens':
        return pd.Series(res).apply(lambda t: ' '.join(t))
    elif strat == 'strings':
        return pd.Series(res).apply(lambda t: '_'.join(t))
