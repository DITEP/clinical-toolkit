"""
This script is intended to show the usage of word2vec for
our use-case
"""
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import word2vec
from sklearn.manifold import TSNE
from preprocessing.html_parser.parser import ReportsParser


path = '../data/reports.csv'

df = pd.read_csv(path, sep=';').head(1000)
parser = ReportsParser(strategy='tokens')
X = parser.transform(df)

w2v = word2vec.Word2Vec(X,
                        size=128,
                        window=8,
                        sample=0.1)

word_vectors = w2v[w2v.wv.vocab]
words = w2v.wv.vocab

words_embedded = TSNE().fit_transform(word_vectors)
plt.scatter(words_embedded[:, 0], words_embedded[:, 1], s=10)

for i, word in enumerate(words):
    plt.annotate(word, xy=(words_embedded[i, 0], words_embedded[i, 1]))
plt.show()

