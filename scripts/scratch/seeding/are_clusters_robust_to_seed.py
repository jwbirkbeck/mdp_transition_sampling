import os, glob
import pandas as pd
import numpy as np
from src.utils.consts import task_pool_10
from src.utils.filepaths import *
import kmedoids
import pickle


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

def test_seed(cluster_size, seed):
    with open(os.path.join(results_path_local, 'agent_comparison', f'cluster_info_{cluster_size}.pkl'), 'rb') as file:
        cluster_info = pickle.load(file)
        saved_clusters = cluster_info['clusters']

    km = kmedoids.KMedoids(cluster_size, method='fasterpam', random_state=seed)
    c = km.fit(dist_matrix)
    new_clusters = c.labels_

    for old_label in np.unique(saved_clusters):
        old_locations = np.where(saved_clusters == old_label)[0]
        example_location = old_locations[0]
        new_label = new_clusters[example_location]
        new_locations = np.where(new_clusters == new_label)[0]
        if old_locations.shape[0] != new_locations.shape[0]:
            return False
        if not np.all(old_locations == new_locations):
            return False
        else:
            pass
    return True

# 350000
for seed in range(1000000):
    if seed % 5000 == 0:
        print(f'step: {seed}')
    clusters_match = []
    restart = False
    for cluster_size in [2, 4, 6, 8, 10]:
        test_val = test_seed(cluster_size, seed)
        clusters_match.append(test_val)
        if not test_val:
            restart = True
            break
    if not restart and np.all(clusters_match):
        print(f'seed found: {seed}')
        break



cluster_size = 10
with open(os.path.join(results_path_local, 'agent_comparison', f'cluster_info_{cluster_size}.pkl'), 'rb') as file:
    cluster_info = pickle.load(file)
    saved_clusters = cluster_info['clusters']

km = kmedoids.KMedoids(cluster_size, method='fasterpam', random_state=4)
c = km.fit(dist_matrix)
new_clusters = c.labels_

print(saved_clusters)
print(new_clusters)