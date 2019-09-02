import numpy as np
import ensemble_bag_deepscoring as ds
from sklearn.metrics import accuracy_score
import itertools

nbr_clf = 6
nbr_obs = 10000
nbr_class = 3
preds = np.random.rand(nbr_clf, nbr_obs, nbr_class)
truth = np.random.randint(nbr_class,size=nbr_obs)
perf = accuracy_score

preds = ds.expansion(preds)
print("large_preds",preds.shape)
preds = ds.reduction(preds, truth, perf,nbr_to_select=10, filter="HH")
print("filtered_preds",preds.shape)
preds = ds.expansion(preds)
print("large_preds",preds.shape)
preds = ds.reduction(preds, truth, perf,nbr_to_select=8, filter="HH")
print("filtered_preds",preds.shape)
preds = ds.expansion(preds)
print("large_preds",preds.shape)
preds = ds.reduction(preds, truth, perf,nbr_to_select=1, filter="perf")
print("filtered_preds",preds.shape)
