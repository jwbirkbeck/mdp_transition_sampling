import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10
np.set_printoptions(suppress=True)

def get_job_infos(tasks_a, tasks_b, repeats):
    job_configs = []
    for task_a in tasks_a:
        for task_b in tasks_b:
            job_configs.append({'task_a': [task_a], 'task_b': [task_b], 'repeats': [repeats]})
    return job_configs


def mp_calculate_distances(job_config):
    task_a = job_config['task_a'][0]
    task_b = job_config['task_b'][0]
    repeats = job_config['repeats'][0]
    w1_dists = pd.DataFrame()
    for rep in range(repeats):
        env_a = MetaWorldWrapper(task_name=task_a)
        env_a.change_task(task_name=task_a, task_number=0)
        env_a.reset()

        env_b = MetaWorldWrapper(task_name=task_b)
        env_b.change_task(task_name=task_b, task_number=0)
        env_b.reset()

        sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                       state_space=env_a.observation_space, action_space=env_a.action_space)

        dist = sampler.get_difference(n_states=50, n_transitions=1)
        pd_row = pd.DataFrame({'task_a': [task_a], 'task_b': [task_b], 'rep': [rep], 'w1': [dist]})
        w1_dists = pd.concat((w1_dists, pd_row))
    return w1_dists


job_configs = get_job_infos(tasks_a=task_pool_10, tasks_b=task_pool_10, repeats=5)

with mp.Pool(14) as pool:
    w1_dists_tasks = pd.concat(pool.map(mp_calculate_distances, job_configs))
print('done')

# with open('mw_w1_dists_tasks.pkl', 'wb') as file:
#     pickle.dump(w1_dists_tasks, file)

dists2 = w1_dists_tasks.drop(['rep'], axis=1)
dist_matrix = dists2.sort_values(by=['task_a', 'task_b']).groupby(['task_a', 'task_b']).mean().reset_index().pivot(index='task_a', columns='task_b', values='w1')
dist_matrix = np.array(dist_matrix).tolist()

import matplotlib.pyplot as plt
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
# plt.savefig('mw_dist_matrix.png', dpi=300)
plt.show()