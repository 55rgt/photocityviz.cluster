from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
import json
import random

NUM_OF_CLUSTERS = 10
FILE_OUTPUT_NAME = 'Agglomerative'
ITERATION = 3
DIVISION = 100

accData = {}

with open('./data/Total_refined_short_10.json') as json_file:
    json_data = json.load(json_file)

for n in range(ITERATION):
    # randomize data
    random.shuffle(json_data)
    # make matrix
    nameList = []
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for datum in json_data:
        nameList.append(datum['name'])
        for label in datum['labels_m']:
            index = vocabulary.setdefault(label, len(vocabulary))
            indices.append(index)
            data.append(datum['labels_m'][label])
        indptr.append(len(indices))
    X = csr_matrix((data, indices, indptr), dtype=float).toarray()
    if n == 0:
        for name in nameList:
            accData[name] = []
    AC = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS)
    lenX = len(X)
    summation = 0
    result_total = []
    for i in range(DIVISION):
        start = int(lenX * i / DIVISION)
        end = int(lenX * (i + 1) / DIVISION)
        result_trial = AC.fit_predict(X[start: end])
        # print(len(result))
        result_total.extend(result_trial)
        summation += len(result_trial)
        if i == 1:
            print(result_trial)
        # fit_predict 얘가 지금 1000 result  근데 데이터 1000
        # print(result)
    print(summation)




# with open('./output/{}_clusters_{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
#     json.dump(AC.labels_.tolist(), outfile)


