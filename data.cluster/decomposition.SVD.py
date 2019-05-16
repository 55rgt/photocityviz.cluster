from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy.sparse import csr_matrix

import json

with open('../data/Total_refined_sample_short.json') as file:
    jsonData = json.load(file)

nameList = []

indptr = [0]
indices = []
data = []
vocabulary = {}

for datum in jsonData:
    for label in datum['labels']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float)

svd = TruncatedSVD(n_components=2, n_iter=100, random_state=42)
svd.fit(X)

print(svd)
print(svd.singular_values_)
print(svd.explained_variance_ratio_)


