from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import numpy
import json
import time

with open('../data/Total_cluster_short.json') as file:
    jsonData = json.load(file)

nameList = []

indptr = [0]
indices = []
data = []
vocabulary = {}

for datum in jsonData:
    nameList.append(datum['name'])
    for label in datum['labels']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float)
svd = TruncatedSVD(n_components=27, n_iter=10, random_state=42)
svd.fit(X)
print(svd.explained_variance_ratio_.sum())

t = svd.transform(X)

tSVD = numpy.around(t, decimals=3)


time_start = time.time()

TSNE = TSNE(n_components=2)

TSNE_result = TSNE.fit_transform(tSVD)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

result = numpy.around(TSNE_result, decimals=3)

print(result)

with open('../output/TSNE.json', 'w') as outfile:
    json.dump(result.tolist(), outfile)

with open('../output/TSVD.json', 'w') as outfile:
    json.dump(tSVD.tolist(), outfile)
