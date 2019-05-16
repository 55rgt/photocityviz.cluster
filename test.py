from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

rossi = load_rossi()

print(rossi)

cph = CoxPHFitter()
cph.fit(rossi, duration_col="week", event_col="arrest")

print(cph)

# Three ways to view the c-index:
# method one

# method two
print(cph.score_)

# method three
from lifelines.utils import concordance_index
# print(concordance_index(rossi['week'], -cph.predict_partial_hazard(rossi), rossi['arrest']))