from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
import json

NUM_OF_CLUSTERS = 10
FILE_OUTPUT_NAME = 'agglomerative'

with open('./data/Total_refined_short.json') as json_file:
    json_data = json.load(json_file)

nameList = []

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

AC = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS)

lenX = len(X)

sum = 0

for i in range(100):
    start = int(lenX * i / 100)
    end = int(lenX * (i + 1) / 100)
    # print(start, end)
    result = AC.fit_predict(X[start: end])
    print(len(result))
    sum += len(result)
    # fit_predict 얘가 지금 1000 result  근데 데이터 1000
    # print(result)
print(sum)

# with open('./output/{}_clusters_{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
#     json.dump(AC.labels_.tolist(), outfile)
