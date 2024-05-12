import glob
import os
import kmedoids
import pickle
import numpy as np
import pandas as pd
from src.utils.consts import task_pool_10

mapping_save_dir = '/opt/project/src/policy_selector/'

results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
w1_dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    w1_dists = pd.concat((w1_dists, tmp))
w1_dists.reset_index()

results2 = w1_dists[w1_dists.env_a.isin(task_pool_10)]
results2 = results2[results2.env_b.isin(task_pool_10)]
results2 = results2.query('n_states==25 and n_transitions==10')
results2 = results2.drop(['n_states', 'n_transitions'], axis=1)

dist_matrix = (results2.sort_values(by = ['env_a', 'env_b']).
               groupby(['env_a', 'env_b']).median().reset_index().
               pivot(index='env_a', columns='env_b', values='dist'))
dist_matrix = np.array(dist_matrix).tolist()

n_clusters = 3
km = kmedoids.KMedoids(n_clusters, method='fasterpam')
c = km.fit(dist_matrix)
mapping_3 = np.array([task_pool_10, c.labels_], dtype='object').transpose()
with open(os.path.join(mapping_save_dir, 'mapping_3.pickle'), 'wb') as file:
    pickle.dump(mapping_3, file)

n_clusters = 6
km = kmedoids.KMedoids(n_clusters, method='fasterpam')
c = km.fit(dist_matrix)
mapping_6 = np.array([task_pool_10, c.labels_], dtype='object').transpose()
with open(os.path.join(mapping_save_dir, 'mapping_6.pickle'), 'wb') as file:
    pickle.dump(mapping_6, file)

n_clusters = 10
km = kmedoids.KMedoids(n_clusters, method='fasterpam')
c = km.fit(dist_matrix)
mapping_10 = np.array([task_pool_10, c.labels_], dtype='object').transpose()
with open(os.path.join(mapping_save_dir, 'mapping_10.pickle'), 'wb') as file:
    pickle.dump(mapping_10, file)


mapping_3r = np.array([task_pool_10, np.random.choice([0, 1, 2], size=10, replace=True)], dtype=object).transpose()
mapping_6r = np.array([task_pool_10, np.random.choice([0, 1, 2, 3, 4, 5], size=10, replace=True)], dtype=object).transpose()

with open(os.path.join(mapping_save_dir, 'mapping_3r.pickle'), 'wb') as file:
    pickle.dump(mapping_3r, file)

with open(os.path.join(mapping_save_dir, 'mapping_6r.pickle'), 'wb') as file:
    pickle.dump(mapping_6r, file)