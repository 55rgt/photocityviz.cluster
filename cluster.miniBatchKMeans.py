from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import json

NUM_OF_CLUSTERS = 12
FILE_OUTPUT_NAME = 'miniBatchKMeans'

with open('./data/Total_refined_short_10.json') as json_file:
    json_data = json.load(json_file)

nameList = []

indptr = [0]
indices = []
data = []
vocabulary = {}
for datum in json_data:
    # print(datum['name'])  # 파일명
    # print(datum['labels'])  # 라벨들
    for label in datum['labels_m']:
        index = vocabulary.setdefault(label, len(vocabulary))
        indices.append(index)
        data.append(datum['labels_m'][label])
    indptr.append(len(indices))

X = csr_matrix((data, indices, indptr), dtype=float).toarray()

print(X)


miniBatchKMeans = MiniBatchKMeans(n_clusters=NUM_OF_CLUSTERS, n_init=5, random_state=0).fit(X)
# print(miniBatchKMeans.cluster_centers_.tolist())
# print(miniBatchKMeans.inertia_.tolist())
print(miniBatchKMeans.labels_.tolist())

# with open('./output/{}_clusters__{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
#     json.dump(miniBatchKMeans.labels_.tolist(), outfile)
