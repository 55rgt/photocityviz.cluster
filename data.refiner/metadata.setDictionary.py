from wordsegment import load, segment
import json

load()

dic = []
dic_map = {}

with open('./data/metadata_counter.json') as file:
    data = json.load(file)

for key in data.keys():
    segments = segment(key)
    if key not in dic_map:
        dic_map[key] = segments
        dic.append(key)

dic.sort(key=len)

with open('./output/metadata_dictionary_map.json', 'w') as outfile:
    json.dump(dic_map, outfile)

with open('./output/metadata_dictionary.json', 'w') as outfile:
    json.dump(dic, outfile)
