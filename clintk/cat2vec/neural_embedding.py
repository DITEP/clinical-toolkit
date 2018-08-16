"""
Embedding high cardinality categorical variables with distributed
representations

The first embedder relies on `Word2Vec` algorithm to learn vector
representations of words in a corpus

.. [1] "Distributed Representations of Words and Phrases and their
  Compositionality", Mikolov et al, Advances in Neural Information Processing
  Systems 26, pp 3111--3119, 2013.


The second one is based on `transfer learning
<https://en.wikipedia.org/wiki/Transfer_learning>`_ : we train a fully
connected neural network on a predictive task (only supports binary
classification for now) so that the upper layers learn higher level
representations of the categories.
After training, we can extract the categories vectors in the embedding space
"""
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from sklearn.base import BaseEstimator
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout


from ..text2vec import tools


class W2VVectorizer(object):
    """ vectorizes categories with word2vec model

    @deprecated

    Parameters
    ----------
    group_key : str
        name of the column to group

    category_col : str
        name of the column containing the categorical variables

    size : int, default=128
        dimension of the embedding vector

    min_count : int, default=1
        minimum amount of instances to integrate it to the model

    sg : int {0, 1}, default=1
        0 for skip-gram word2vec model
        1 for CBOW (best suited for small datasets)

    window : int, default=3
        size of the context

    strategy : str {'tokens', 'strings'}, default='tokens'
        if 'tokens', categories containing several words are split
        else, each category is considered as a word

            """
    def __init__(self, group_key, category_col,
                 size=128, min_count=1, sg=1,
                 window=3, strategy='tokens',
                 seed=0):
        self.key = group_key,
        self.cat_col = category_col
        self.size = size
        self.min_count = min_count
        self.sg = sg
        self.window = window
        self.strategy = strategy
        self.seed = seed
        self.w2v_ = None

    def fit(self, X, y=None):
        """ fits the model by grouping categories by group_key in order to
        embed categories as text

        Parameters
        ----------
        X : pd.DataFrame
        y

        Returns
        -------

        """
        df_grouped = X.groupby(self.key).agg({self.cat_col: ' '.join})
        df_grouped[self.cat_col] = df_grouped[self.cat_col] \
            .apply(lambda s: s.split(' '))

        self.w2v_ = Word2Vec(df_grouped[self.cat_col],
                             size=self.size,
                             window=self.window,
                             min_count=self.min_count,
                             sg=self.sg,
                             seed=self.seed).wv
        return self

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : pd.DataFrame

        y : None

        Returns
        -------

        """
        categories = X[self.cat_col].apply(lambda s: s.split(' '))

        vectors = categories.apply(lambda cat: tools.avg_document(self.w2v_, cat))

        X[self.cat_col + '_embedded'] = vectors

        return X

    def fit_pretrained(self, path, **kwargs):
        """
        fits model using pretrained word embedding from
        https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md


        Parameters
        ----------
        path : str
            path do wiki.lg.vec file

        Returns
        -------

        """
        self.w2v_ = KeyedVectors.load_word2vec_format(path, **kwargs)

        return self


class NeuralEmbedder(BaseEstimator):
    """ Trains a MLP classifier to learn a distributed representation of
    categories

    Only available for binary targets

    @TODO optimizer argument should be able to receive keras.Optimizer class
    @TODO + batch_size + validation set ?

    input_dim : tuple, (int, int)
        input_dim[0] number of units in inpuot layer
        input_dim[1] : dimension of the input layer (= number of features)

    layers : tuple
        The ith element represents the number of neurons in the ith hidden
        layer. Similar to sklearn's MLP

    activation : str, default='relu'
        activation function in the intermediate layers

    output : str, default='sigmoid'
        output activation function, only supports sigmoid for binary
        classification

    optimizer : str, default='adam'
        optimizing function for backpropagation
        check https://keras.io/optimizers for available algorithms

    loss : str, default='binary-crossentropy'
        loss computed for optimization
        check https://keras.io/losses

    dropout : str, default=0.5
        dropout rate

    metrics : list, default=['acc', 'mae']
        metrics used uring training and testing

    epochs : int, default=20
        number of epochs


    """
    def __init__(self, input_dim, layers,
                 activation='relu', output='sigmoid',
                 optimizer='adam',
                 loss='binary-crossentropy',
                 dropout=0.5,
                 metrics=['acc', 'mae'],
                 epochs=20):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs

        # indicator of training state
        self.fit_ = None

        self.model = Sequential()

        # input layer
        self.model.add(Dense(input_dim[0],
                             activation=activation,
                             input_dim=input_dim[1]))
        self.model.add(Dropout(dropout))

        # stacking following layers
        for i, units in enumerate(layers):
            self.model.add(Dense(units, activation=activation))
            self.model.add(Dropout(dropout))

        self.model.add(Dense(1, activation=output))

    def fit(self, X, y):
        """ trains the model using input data

        Parameters
        ----------
        X : iterable
            feature matrix

        y : iterable
            target vector (possibly one-hot-encoded?)

        Returns
        -------
        keras.History.history
            record of training loss values and metrics values at successive
            epochs, as well as validation loss values and validation metrics
            values (if applicable)

        """
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        hist = self.model.fit(X, y, epochs=self.epochs)
        self.fit_ = True

        return hist

    def transform(self, X):
        """ Transform X into a distributed representation learned by fit

        Parameters
        ----------
        X : iterable
            feature matrix to embed

        Returns
        -------
        numpy array
            X projected into an embedding space

        """
        model_cut = clone_model(self.model)

        # removing output layer + last dropout
        # @TODO change method (sub optimal and inelegant)
        model_cut.pop()
        model_cut.pop()

        return model_cut.predict(X)
