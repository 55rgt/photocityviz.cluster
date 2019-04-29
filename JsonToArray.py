from scipy.sparse import csr_matrix
import json

# JSON 라벨 파일을 쓸 수 있는 Matrix로 만든다.

# [[1 1 1 ... 0 0 0]
#  [0 1 0 ... 0 0 0]
#  [0 1 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 1 1 1]]


NUM_OF_CLUSTERS = 4
FILE_OUTPUT_NAME = 'sample'

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

with open('./data/label_label.json', 'w') as outfile:
    json.dump(X.tolist(), outfile)
