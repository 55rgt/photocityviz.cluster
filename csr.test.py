from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import json

NUM_OF_CLUSTERS = 10

with open('./data/sample.json') as json_file:
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

kmeans = KMeans(n_clusters=NUM_OF_CLUSTERS, n_init=5, algorithm="full", random_state=0).fit(X)
agglomerative = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS).fit(X)
spectral = SpectralClustering(n_clusters=NUM_OF_CLUSTERS, assign_labels="discretize", random_state=0).fit(X)

print(kmeans.labels_)
print(agglomerative.labels_)
print(spectral.labels_)
