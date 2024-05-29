import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.utils.consts import task_pool, task_pool_10
import networkx as nx
import numpy as np
import kmedoids


results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
w1_dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    w1_dists = pd.concat((w1_dists, tmp))
w1_dists.reset_index()


# task_selection = task_pool[0:10]
task_selection = task_pool_10
# task_selection = ['reach-v2', 'reach-wall-v2', 'handle-press-v2', 'handle-press-side-v2']
trained_agents_pool = task_selection

results2 = w1_dists[w1_dists.env_a.isin(trained_agents_pool)]
results2 = results2[results2.env_b.isin(trained_agents_pool)]
results2 = results2.query('n_states==25 and n_transitions==10')
results2 = results2.drop(['n_states', 'n_transitions'], axis=1)

dist_matrix = results2.groupby(['env_a', 'env_b']).median().reset_index().pivot(index='env_a', columns='env_b', values='dist')
dist_matrix = np.array(dist_matrix).tolist()

tmp = pd.DataFrame(dist_matrix)
fig, ax = plt.subplots()
im = ax.imshow(tmp)
plt.xlabel("MDP index")
plt.ylabel("MDP index")
ax.set_xticks(range(0, 10), labels=range(0, 10))
ax.set_yticks(range(0, 10), labels=range(0, 10))

# Loop over data dimensions and create text annotations.
for i in range(0, 10):
    for j in range(0, 10):
        text = ax.text(j, i, round(dist_matrix[i][j], 2),
                       ha="center", va="center", color="black", size=8)
plt.title("10-MDP Wasserstein distance matrix")
plt.tight_layout()
plt.savefig('dist_matrix.png', dpi=200)
plt.show()

G = nx.Graph()

km = kmedoids.KMedoids(3, method='fasterpam')
c = km.fit(dist_matrix)
print("Loss is:", c.inertia_)
clusters = c.labels_

max_dist = max(results2.dist)
labels = {}
for n in range(len(dist_matrix)):
    for m in range(len(dist_matrix)-(n+1)):
        G.add_edge(n,n+m+1, edge_length=max_dist - dist_matrix[n][n+m+1])
        labels[ (n,n+m+1) ] = str(round(dist_matrix[n][n+m+1], 2))

pos=nx.spring_layout(G, weight='edge_length', iterations=1000)
nx.draw(G, pos, node_color=clusters, edge_color='lightgrey', with_labels=True, node_size=1200)
plt.xlim([0.93 * i for i in plt.xlim()])
plt.ylim([0.93 * i for i in plt.ylim()])
plt.savefig('clusters.png', dpi=200, bbox_inches="tight")
plt.show()


kmin, kmax = 1, len(task_selection)
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
plt.savefig('cluster_by_score.png', dpi=200)
plt.show()

kmin, kmax = 1, len(10)
dm = kmedoids.dynmsc(dist_matrix, kmax, kmin)
print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
print("Medoid Silhouette over range of k:", dm.losses)
print("Range of k:", dm.rangek)


plt.scatter(dm.rangek, dm.losses)
plt.show()