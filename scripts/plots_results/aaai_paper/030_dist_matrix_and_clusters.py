import os, glob
import pandas as pd
import numpy as np
from src.utils.consts import task_pool_10
from src.utils.filepaths import *
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from src.utils.filepaths import results_path_iridis, results_path_local

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

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(dist_matrix)
plt.xlabel("MDP index")
plt.ylabel("MDP index")
ax.set_xticks(range(0, 10), labels=range(0, 10))
ax.set_yticks(range(0, 10), labels=range(0, 10))
# Loop over data dimensions and create text annotations.
for i in range(0, 10):
    for j in range(0, 10):
        text = ax.text(j, i, round(dist_matrix[i][j], 2),
                       ha="center", va="center", size=8)
plt.title("Distance matrix for 10 MetaWorld MDPs")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('mw_dist_matrix.png', dpi=300)
plt.show()

cluster_size = 4

with open(os.path.join(results_path_local, 'agent_comparison' ,f'cluster_info_{cluster_size}.pkl'), 'rb') as file:
    cluster_info = pickle.load(file)
clusters = cluster_info['clusters']

G = nx.Graph()

max_dist = max(dists2.dist)
labels = {}
for n in range(len(dist_matrix)):
    for m in range(len(dist_matrix)-(n+1)):
        G.add_edge(n,n+m+1, edge_length=max_dist - dist_matrix[n][n+m+1])
        labels[ (n,n+m+1) ] = str(round(dist_matrix[n][n+m+1], 2))

pos=nx.spring_layout(G, weight='edge_length', iterations=1000)
nx.draw(G, pos, node_color=clusters, edge_color='lightgrey', with_labels=True, node_size=1200)
plt.xlim([0.93 * i for i in plt.xlim()])
plt.ylim([0.93 * i for i in plt.ylim()])
plt.savefig('mw_clusters.png', dpi=300, bbox_inches="tight")
plt.show()

import kmedoids
kmin, kmax = 1, len(task_pool_10)
dm = kmedoids.dynmsc(dist_matrix, kmax, kmin)
print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
print("Medoid Silhouette over range of k:", dm.losses)
print("Range of k:", dm.rangek)

plt.figure(figsize=(9/1.4, 6/1.4))
plt.scatter(dm.rangek, dm.losses)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette ccore")
plt.title("K-medoid silhouette score (higher is better)")
plt.tight_layout()
plt.xticks(ticks=range(11), labels=range(11))
plt.yticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# plt.savefig('cluster_by_score.png', dpi=200)
plt.show()
