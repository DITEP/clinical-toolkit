"""
sample script for categorical encoding
"""

import re
import unidecode
import pandas as pd

from reptk.features import transform


def normalize_cat(X, strat='tokens'):
    """ normalize categories in a

    Parameters
    ----------
    X : iterable

    Returns
    -------

    """
    res = []
    for x in X:
        x = unidecode.unidecode(x)
        patt = re.compile('[\W_]+')
        x_norm = transform.text_normalize(patt.sub(' ', x), ['sw'])
        res.append(x_norm)
    if strat == 'tokens':
        return pd.Series(res).apply(lambda t: ' '.join(t))
    elif strat == 'strings':
        return pd.Series(res).apply(lambda t: '_'.join(t))



