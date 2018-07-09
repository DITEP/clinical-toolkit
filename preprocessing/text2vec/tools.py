"""

"""
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer


def avg_document(model, document):
    """ computes the average vector of the words in document
    in the word2vec model space

    Parameters
    ----------
    model : word2vec.KeyedVectors instance
    document : list
        tokenized document to transform into a vector

    Returns
    -------
    avg : np.ndarray
        the average of all the words in document

    """
    vocab = model.vocab
    n_features = model.vector_size # change to model.vector_sizes
    count = 0

    vectors = np.zeros((n_features,), dtype='float64')

    for word in document:
        if word in vocab:
            new_vec = model[word]
            # print(new_vec.shape, vectors.shape)    #debug statement
            vectors = np.vstack((vectors, new_vec))
            count += 1
    # print(vectors.shape)
    if count > 0:
        avg = np.mean(vectors[1:], axis=0)
    else:
        avg = vectors

    return avg


def avg_corpus(model, corpus):
    """ computes average vector for each document of the corpus

    Parameters
    ----------
    model : gensim.word2vec.Word2Vec instance
        Trained word2vec model
    corpus : iterable of iterables

    Returns
    -------
    """
    # n, p = len(corpus), model.layer1_size
    features = [avg_document(model, doc) for doc in corpus]

    return np.array(features)


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