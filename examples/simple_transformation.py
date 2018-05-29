import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.text2vec.transformers import HTMLParser, Text2Vector
from sklearn.manifold import TSNE

sns.set()

# load data
path = 'data/reports.csv'
df = pd.read_csv(path, sep=';', encoding='utf-8') .head(5000)

# parse the reports
parser = HTMLParser(strategy='tokens')
X = parser.transform(df)

# vectorize the text
text2vec = Text2Vector().fit(X)
vectorized_x = text2vec.transform(X)

# plot the result using t-sne reduction
X_embedded = TSNE().fit_transform(vectorized_x)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10)
plt.show()

