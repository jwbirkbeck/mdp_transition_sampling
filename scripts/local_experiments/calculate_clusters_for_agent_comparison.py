import os, glob
import pandas as pd
import numpy as np
from src.utils.consts import task_pool_10
from src.utils.filepaths import *
import kmedoids
import pickle
from src.utils.filepaths import results_path_iridis

# # #
# Calculate the task policy mapping
# # #

distances_path = os.path.join(results_path_local, 'task_transition_sampling/')
all_filenames = glob.glob(distances_path + "results_[0-9]*.csv")
dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    dists = pd.concat((dists, tmp))
dists.reset_index()

dists2 = dists[dists.env_a.isin(task_pool_10)]
dists2 = dists2[dists2.env_b.isin(task_pool_10)]
dists2 = dists2.query('n_states==25 and n_transitions==10')
dists2 = dists2.drop(['n_states', 'n_transitions'], axis=1)

dist_matrix = dists2.sort_values(by=['env_a', 'env_b']).groupby(['env_a', 'env_b']).median().reset_index().pivot(index='env_a', columns='env_b', values='dist')
dist_matrix = np.array(dist_matrix).tolist()

for cluster_size in [2, 4, 6, 8, 10]:
    km = kmedoids.KMedoids(cluster_size, method='fasterpam')
    c = km.fit(dist_matrix)
    clusters = c.labels_
    mapping = np.array([task_pool_10, clusters], dtype='object').transpose()
    cluster_info = {'clusters': clusters,
                    'mapping': mapping}
    with open(os.path.join(results_path_local, 'agent_comparison' ,f'cluster_info_{cluster_size}.pkl'), 'wb') as file:
        pickle.dump(cluster_info, file)
