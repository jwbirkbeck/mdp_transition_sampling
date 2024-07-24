import multiprocessing as mp
import numpy as np
import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10, ns_test_inds
from src.utils.funcs import get_standard_ns_dist
np.set_printoptions(suppress=True)

def get_job_infos(tasks, ns_test_inds, repeats):
    job_configs = []
    for task in tasks:
        for ind in ns_test_inds:
            job_configs.append({'task': [task], 'ns_test_ind': [ind], 'repeats': [repeats]})
    return job_configs


# def mp_calculate_distances(job_config):
#     task = job_config['task'][0]
#     ns_test_ind = job_config['ns_test_ind'][0]
#     repeats = job_config['repeats'][0]
#     w1_dists = pd.DataFrame()
#     for rep in range(repeats):
#         env_a = MetaWorldWrapper(task_name=task)
#         env_a.change_task(task_name=task)
#         env_a.reset()
#
#         env_b = MetaWorldWrapper(task_name=task)
#         env_b.change_task(task_name=task)
#         env_b.reset()
#
#         ns_dist = get_standard_ns_dist(env=env_a)  # env_a == env_b
#         ns_dist.freeze()
#         env_b.ns_dist = ns_dist
#
#         env_b.ns_dist.set_sequence_ind(ind=ns_test_ind)
#
#         sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
#                                        state_space=env_a.observation_space, action_space=env_a.action_space)
#
#         dist = sampler.get_difference(n_states=50, n_transitions=1)
#         pd_row = pd.DataFrame({'task': [task], 'ns_test_ind': [ns_test_ind], 'rep': [rep], 'w1': [dist]})
#         w1_dists = pd.concat((w1_dists, pd_row))
#         return w1_dists
#
#
# job_configs = get_job_infos(tasks=task_pool_10, ns_test_inds=ns_test_inds, repeats=5)
#
# with mp.Pool(14) as pool:
#     w1_dists = pd.concat(pool.map(mp_calculate_distances, job_configs))
# print('done')
#
# w1_dists.to_csv('ns_w1_dists.csv')

import matplotlib.pyplot as plt
w1_dists.groupby(['ns_test_ind'])['w1'].boxplot()
plt.show()

fig, ax = plt.subplots()
for ind in w1_dists.ns_test_ind.unique():
    plotdata = w1_dists[w1_dists.ns_test_ind==ind]
    err = np.percentile(plotdata.w1, [25, 75])
    plt.errorbar(x=ind, y=plotdata.w1.median(), yerr=err[1] - err[0], capsize=2)
ax.set_xticks(ticks=np.arange(0, 100001, 10000), labels=np.arange(0, 100001, 10000))
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()
