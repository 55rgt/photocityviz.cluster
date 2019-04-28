from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
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

print('Before1')
print('Before2')

X_train, X_test = train_test_split(X, test_size=0.1, train_size=0.1, random_state=42)

print('After')

AC = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS)

print('AC')

AC.fit(X_train)

print('After fitting')
labels = AC.labels_
print(labels)

# KN = KNeighborsClassifier()
# KN.fit(X_train, labels)
# labels2 = KN.predict(X)
#
# print(labels2)

# agglomerative = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS).fit(X)
# spectral = SpectralClustering(n_clusters=NUM_OF_CLUSTERS, assign_labels="discretize", random_state=0).fit(X)
#
# print(miniBatchKMeans.labels_)
# print(agglomerative.labels_)
# print(spectral.labels_)