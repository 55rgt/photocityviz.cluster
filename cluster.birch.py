from scipy.sparse import csr_matrix
from sklearn.cluster import Birch
import json

NUM_OF_CLUSTERS = 4
FILE_OUTPUT_NAME = 'birch'

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

brc = Birch(n_clusters=NUM_OF_CLUSTERS).fit(X)

with open('./output/{}_clusters_{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
    json.dump(brc.labels_.tolist(), outfile)
