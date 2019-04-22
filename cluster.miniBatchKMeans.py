from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import json

NUM_OF_CLUSTERS = 10

with open('./data/Total_labels_short_refined.json') as json_file:
    json_data = json.load(json_file)
indptr = [0]
indices = []
data = []
vocabulary = {}
for d in json_data:
    for term in d:
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=int).toarray()

print(X)


miniBatchKMeans = MiniBatchKMeans(n_clusters=NUM_OF_CLUSTERS, n_init=5, random_state=0).fit(X)
with open('./output/miniBatchKMeans_clusters_{}.json'.format(NUM_OF_CLUSTERS), 'w') as outfile:
    json.dump(miniBatchKMeans.labels_.tolist(), outfile)
