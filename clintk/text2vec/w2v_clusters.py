"""
clustering of word embeddings

@TODO documentation of the module
"""
import numpy as np

from sklearn.base import BaseEstimator
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class WordClustering(BaseEstimator):
    """ theme-affinity vectorization of documents

    w2v_size : int, default=128
       size of the hidden layer in the embedding Word2Vec model

    n_clusters : int, default=30
        number of clusters,  to the number of output parameters for the
        vectorization.
        It is advised to set `n_clusters` to the approximate number of
        lexical fields

    pca : sklearn.decomposition instance, default=PCA(n_components=0.9)
        reduction of dimension of the embeddings before clustering

    clustering : sklearn.cluster instace, default=KMeans(n_clusters=30)
        clustering algorithm
        The number of clusters must be equal to `n_clusters`

    """
    def __init__(self,
                 w2v_size=128,
                 n_clusters=30,
                 pca=PCA(n_components=0.9),
                 clustering=KMeans(n_clusters=30)):
        self.w2v_size = w2v_size
        self.n_clusters = n_clusters
        self.pca = pca
        self.clustering = clustering

        # vocabulary
        self.vocabulary_ = None
        # distribued representation of the words
        self.word_vectors_ = None
        # cluster id for each word
        self.cluster_ids_ = None

        self.clustering.set_params(n_clusters=n_clusters)


    def fit(self, X, y=None, **fit_params):
        """ train w2v and clustering models

        Parameters
        ----------
        X : iterable of iterable
            corpus of tokenized documents

        y : None

        fit_params : additionnal parameters for word2vec algorithm

        Returns
        -------
        self

        """
        w2v = Word2Vec(X, size=self.w2v_size)

        self.vocabulary_ = w2v.wv.vocab
        self.word_vectors_ = w2v[self.vocabulary_]

        pca_word_vectors = self.pca.fit_transform(self.word_vectors_)

        self.cluster_ids_ = self.clustering.fit_predict(pca_word_vectors)

        return self

    def transform(self, X, y=None):
        """ transforms each row of `X` into a vector of clusters affinities

        Parameters
        ----------
        X : iterable of iterable

        y: None

        Returns
        -------
        numpy.ndarray, shape=(n, p)
            transformed docments, where `p=n_cluster`

        """
        vectors = []

        for x in X:
            vector = np.zeros(self.n_clusters)
            count = 0
            for t in x:
                try:
                    word_id = self.vocabulary_[t].index
                    word_cluster = self.cluster_ids_[word_id]
                    vector[word_cluster] = vector[word_cluster] + 1
                    count += 1
                # except word is not in vocabular
                except KeyError:
                    pass
            if count > 0:
                vectors.append(vector / count)
            else:
                vectors.append(vector)

        return np.array(vectors)

    def get_clusters_words(self):
        """ return the words in each cluster


        Returns
        -------
        dict
            keys are cluster ids, values are lists of words

        """
        words_cluster = {}
        for cluser_id in np.unique(self.cluster_ids_):
            words_cluster[str(cluser_id)] = []

        for i, word in enumerate(self.vocabulary_):
            label = str(self.cluster_ids_[i])
            words_cluster[label].append(word)

        return words_cluster




def embed_corpus(X, n_clusters, clustering, **kwargs):
    """ transforms X into vector of cluster affinities

    ..deprecated use `WordClustering` object instead
    Parameters
    ----------
    X : iterable of iterable, (length=n)
        corpus of document

    clustering : sklearn.cluster object
        instanciated clustering algorithm

    Returns
    -------
    np.ndarray, shape=(n, n_clusters)

    """
    # fit
    w2v = Word2Vec(X, size=128)

    words = w2v.wv.vocab
    word_vectors = w2v[words]

    pca_word_vectors = PCA(n_components=0.9).fit_transform(word_vectors)

    # clustering = AgglomerativeClustering(n_clusters, affinity='euclidean')

    cluster_ids = clustering.fit_predict(pca_word_vectors)
    
    # transform
    vectors = []
    for x in X:
        vector = np.zeros(n_clusters)
        count = 0
        for t in x:
            try:
                word_id = words[t].index
                word_cluster = cluster_ids[word_id]
                vector[word_cluster] = vector[word_cluster] + 1
                count += 1
            except KeyError:
                pass
        vectors.append(vector / count)

    return np.array(vectors), cluster_ids
