from sklearn import metrics
from scipy.sparse import csr_matrix
import json
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

with open('./data/sample_.json') as json_file:
    json_data = json.load(json_file)

with open('./output/sample_clusters__5.json') as json_file:
    labels = json.load(json_file)

indptr = [0]
indices = []
data = []
vocabulary = {}
for datum in json_data:
    # print(datum['name'])  # 파일명
    # print(datum['labels'])  # 라벨들
    for label in datum['labels']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float).toarray()

print(X)
print(metrics.calinski_harabaz_score(X, labels))
print(metrics.silhouette_score(X, labels, metric='euclidean'))
print(metrics.davies_bouldin_score(X, labels))
