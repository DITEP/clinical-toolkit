"""
Word embedding based on  word2Vec model: turns words into feature
space to input them ino a classification model

"""
# import matplotlib.pyplot as plt
import numpy as np

from gensim.models import word2vec
from sklearn.manifold import t_sne
from sklearn.decomposition import pca


def train_w2v_model(corpus, n_embed=128, window=8, min_word=1,
                    sample=1e-1, sg=1, seed=0, **kwargs):
    """ Trains a Word2Vec model with given parameters
    (useless for now )

    Parameters
    ----------
    corpus
    n_embed
    window
    min_word
    sample
    sg
    seed

    Returns
    -------
    model : word2vec.Word2Vec instance

    """
    return word2vec.Word2Vec(corpus,
                             size=n_embed,
                             window=window,
                             min_count=min_word,
                             sample=sample,
                             sg=sg,
                             seed=seed,
                             *kwargs)


# def plot_w2v(model, s=10, reducer='t-sne'):
#     """ plots a word2vec's model vocab given a dimension reduction
#     algorihm
#
#     Parameters
#     ----------
#     model : word2vec.Word2Vec instance
#         trained w2v model
#     s : int
#         size of the scatter plot
#     reducer : str
#         either 't-sne' for using Stochastic Neighbour Embedding
#         algorithm or 'pca' for Principal Components Analysis
#
#     Returns
#     -------
#     None
#     """
#     if reducer == 't-sne':
#         transformer = t_sne.TSNE(n_components=2)
#     elif reducer == 'pca':
#         transformer = pca.PCA(n_components=2)
#     else:
#         raise ValueError("reducer agrument = {} not in \
#                 ['t-sne', 'pca']".format(reducer))
#
#     words = model.wv.vocab
#     word_vectors = model[words]
#
#     embedded_vect = transformer.fit_transform(word_vectors)
#
#     plt.scatter(embedded_vect[:, 0], embedded_vect[:, 1], s=s)
#     for i, word in enumerate(words):
#         plt.annotate(word, xy=(embedded_vect[i, 0], embedded_vect[i, 1]))
#     plt.show()
#
#     return 'done'


def avg_document(model, document):
    """ computes the average vector of the words in document
    in the word2vec model space

    Parameters
    ----------
    model : word2vec.Word2Vec instance
    document : list
        tokenized document to transform into a vector

    Returns
    -------
    avg : np.ndarray
        the average of all the words in document

    """
    vocab = model.wv.vocab
    n_features = model.layer1_size
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





