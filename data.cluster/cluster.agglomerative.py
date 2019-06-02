from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
import json
import random

NUM_OF_CLUSTERS = 12
FILE_OUTPUT_NAME = 'Agglomerative'
ITERATION = 30
DIVISION = 100

accData = {}

with open('../data/Total_refined_short_10.json') as json_file:
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
    # initialize data if first
    if n == 0:
        for name in nameList:
            accData[name] = []
    # clustering
    AC = AgglomerativeClustering(n_clusters=NUM_OF_CLUSTERS)
    result = AC.fit(X[0:14400])
    print(result.labels_)



    # lenX = len(X)
    # result_total   = []
    # for i in range(DIVISION):
    #     start = int(lenX * i / DIVISION)
    #     end = int(lenX * (i + 1) / DIVISION)
    #     result_trial = AC.fit_predict(X[start: end])
    #     result_total.extend(result_trial)
    # # push data into accData
    # for j in range(len(nameList)):
    #     accData[nameList[j]].append(result_total[j])

# print(accData[list(accData.keys())[0]])







# with open('./output/{}_clusters_{}.json'.format(FILE_OUTPUT_NAME, NUM_OF_CLUSTERS), 'w') as outfile:
#     json.dump(AC.labels_.tolist(), outfile)


