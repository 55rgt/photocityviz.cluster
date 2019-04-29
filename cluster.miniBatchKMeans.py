from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import json

NUM_OF_CLUSTERS = 5
FILE_OUTPUT_NAME = 'miniBatchKMeans'

with open('./data/Total_vision_merged_short_.json') as json_file:
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


miniBatchKMeans = MiniBatchKMeans(n_clusters=NUM_OF_CLUSTERS, n_init=5, random_state=0).fit(X)
print(miniBatchKMeans.cluster_centers_.tolist())
print(miniBatchKMeans.inertia_.tolist())

with open('./output/{}_clusters__{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
    json.dump(miniBatchKMeans.labels_.tolist(), outfile)
