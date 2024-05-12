import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.utils.consts import task_pool, task_pool_10
import networkx as nx
import numpy as np
import kmedoids


results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
w2_dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    w2_dists = pd.concat((w2_dists, tmp))
w2_dists.reset_index()


# task_selection = task_pool[0:10]
task_selection = task_pool_10
# task_selection = ['reach-v2', 'reach-wall-v2', 'handle-press-v2', 'handle-press-side-v2']
trained_agents_pool = task_selection

results2 = w2_dists[w2_dists.env_a.isin(trained_agents_pool)]
results2 = results2[results2.env_b.isin(trained_agents_pool)]
results2 = results2.query('n_states==25 and n_transitions==10')
results2 = results2.drop(['n_states', 'n_transitions'], axis=1)

dist_matrix = results2.groupby(['env_a', 'env_b']).median().reset_index().pivot(index='env_a', columns='env_b', values='dist')
dist_matrix = np.array(dist_matrix).tolist()

G = nx.Graph()

km = kmedoids.KMedoids(2, method='fasterpam')
c = km.fit(dist_matrix)
print("Loss is:", c.inertia_)
clusters = c.labels_

labels = {}
for n in range(len(dist_matrix)):
    for m in range(len(dist_matrix)-(n+1)):
        G.add_edge(n,n+m+1, edge_length=-1.0*dist_matrix[n][n+m+1])
        labels[ (n,n+m+1) ] = str(round(dist_matrix[n][n+m+1], 2))

pos=nx.spring_layout(G, weight='edge_length', iterations=250)
nx.draw(G, pos, node_color=clusters, edge_color='lightgrey')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6)
plt.show()


kmin, kmax = 1, len(task_selection)
dm = kmedoids.dynmsc(dist_matrix, kmax, kmin)
print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
print("Medoid Silhouette over range of k:", dm.losses)
print("Range of k:", dm.rangek)

plt.scatter(dm.rangek, dm.losses)
plt.xlabel("Number of clusters")
plt.ylabel("Medoid silhouette score")
plt.tight_layout()
plt.show()

kmin, kmax = 1, len(10)
dm = kmedoids.dynmsc(dist_matrix, kmax, kmin)
print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
print("Medoid Silhouette over range of k:", dm.losses)
print("Range of k:", dm.rangek)


plt.scatter(dm.rangek, dm.losses)
plt.show()