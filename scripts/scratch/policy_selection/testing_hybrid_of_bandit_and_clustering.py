import glob
import torch
import kmedoids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.policy_selector.simple_policy_selector import SimplePolicySelector
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10

# # # # # # # # # #
# Load wasserstein distances and precompute the clustering
# # # # # # # # # #
results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))
results.reset_index()


results2 = results[results.env_a.isin(task_pool_10)]
results2 = results2[results2.env_b.isin(task_pool_10)]
results2 = results2.query('n_states==25 and n_transitions==10')
results2 = results2.drop(['n_states', 'n_transitions'], axis=1)

dist_matrix = results2.sort_values(by=['env_a', 'env_b']).groupby(['env_a', 'env_b']).median().reset_index().pivot(index='env_a', columns='env_b', values='dist')
dist_matrix = np.array(dist_matrix).tolist()

n_clusters = 3
km = kmedoids.KMedoids(n_clusters, method='fasterpam')
c = km.fit(dist_matrix)
print("Loss is:", c.inertia_)
clusters = c.labels_


device = torch.device('cpu')
mapping = np.array([task_pool_10, clusters], dtype='object').transpose()
env = MetaWorldWrapper(task_name=task_pool_10[0]) # starts at the first task, but probabilistically changes during experiment
policy_selector = SimplePolicySelector(env=env, method='bandit', device=device, task_names=task_pool_10, n_policies=3)
for ind, row in enumerate(policy_selector.task_policy_mapping):
    row[clusters[ind]] = 1.0

p_task_change = 0.1
rewards = []
for _ in range(250):
    print(_)
    if np.random.uniform() < p_task_change:
        new_task_name = np.random.choice(task_pool_10)
        env.change_task(task_name=new_task_name)
    ep_rew, _ = policy_selector.play_episode()
    rewards.append(ep_rew)
print("done")

plt.plot(rewards)
plt.show()

rewards2 = np.array(rewards)
window = 25
plt.plot(np.convolve(rewards2, np.ones(window)/window, mode='valid'))
plt.show()

np.mean(rewards)
