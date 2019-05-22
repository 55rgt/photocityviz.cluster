from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
import json

with open('../data/Total_refined_short_10.json') as file:
    jsonData = json.load(file)

nameList = []

indptr = [0]
indices = []
data = []
vocabulary = {}

for datum in jsonData:
    for label in datum['labels_m']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels_m'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float).toarray()

print(X)





# print(vocabulary) => 이걸로 인덱스 파악 가능

# lenX = len(X)
# for i in range(1000):
#     result = DBSCAN.fit_predict(X[int(lenX * i / 1000): int(lenX * (i + 1) / 1000)])
#     print(result)