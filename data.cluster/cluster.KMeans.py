from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import json

NUM_OF_CLUSTERS = 12
FILE_OUTPUT_NAME = 'KMeans'
ITERATION = 30
DIVISION = 100

accData = {}

with open('../data/Total_cluster_short.json') as json_file:
    json_data = json.load(json_file)

nameList = []

indptr = [0]
indices = []
data = []
vocabulary = {}
for datum in json_data:
    nameList.append(datum['name'])
    for label in datum['labels']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float).toarray()

print(X)

clustering = KMeans(n_clusters=NUM_OF_CLUSTERS, random_state=42).fit(X)


# with open('./output/{}_clusters_{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
#     json.dump(Clustering.labels_.tolist(), outfile)


