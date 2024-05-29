import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metaworld.nonstationarity_distribution import MWNSDistribution
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10, bounded_state_space
np.set_printoptions(suppress=True)

"""
Determine whether all 10 MDPs self-distance is near zero.
Determine why the W1 distances appeared uniform across increasing nonstationarities.
Determine whether the high W1s are legitimate or not.
"""

# self-distances that are large:
# dial-turn-v2

# measuring differences between MDPs:

def get_w1_dists(task_mappings):
    tasks_a = task_mappings['tasks_a']
    tasks_b = task_mappings['tasks_b']
    reps = task_mappings['reps'][0]
    w1_dists = pd.DataFrame()
    for task_a in tasks_a:
        for task_b in tasks_b:
            env_a = MetaWorldWrapper(task_name=task_a)
            env_b = MetaWorldWrapper(task_name=task_b)
            state_space = bounded_state_space
            action_space = env_a.action_space

            sampler = MDPDifferenceSampler(environment_a=env_a,
                                           environment_b=env_b,
                                           state_space=state_space,
                                           action_space=action_space)

            for rep in range(reps):
                env_a.reset()
                env_b.reset()
                dist = sampler.get_difference(n_states=15, n_transitions=2)
                pd_row = pd.DataFrame({'task_a': [task_a], 'task_b': [task_b], 'rep': [rep], 'w1': [dist]})
                w1_dists = pd.concat((w1_dists, pd_row))
    return w1_dists

def make_split_task_mappings(tasks, reps):
    list_of_dicts = []
    for task_a in tasks:
        dict = {'tasks_a': [task_a], 'tasks_b': tasks, 'reps': [reps]}
        list_of_dicts.append(dict)
    return list_of_dicts

#
# list_mappings = make_split_task_mappings(tasks=task_pool_10, reps=3)
#
# with mp.Pool(10) as pool:
#     w1_dists = pd.concat(pool.map(get_w1_dists, list_mappings))
# print('done')
#
# w1_dists = w1_dists.drop('index', axis=1)
#
# for task_a in [task_pool_10[0]]:
#     fig, ax = plt.subplots()
#     for posind, task_b in enumerate(task_pool_10):
#         plotdata = w1_dists.query("task_a=='" + task_a + "' and task_b=='" + task_b + "'")
#         plt.boxplot(plotdata['w1'], positions=[posind])
#     ax.set_xticks(ticks=range(0, posind + 1), labels=task_pool_10)
#     plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
#     plt.tight_layout()
#     plt.show()

def get_w1_dists_ns(job_info):
    tasks = job_info['tasks']
    ns_seq_ind = job_info['ns_seq_ind'][0]
    repeats = job_info['repeats'][0]
    w1_dists = pd.DataFrame()
    for task in tasks:
        env_a = MetaWorldWrapper(task_name=task)
        env_a.change_task(task_name=task, task_number=0)
        env_a.reset()

        env_b = MetaWorldWrapper(task_name=task)
        env_b.change_task(task_name=task, task_number=0)
        env_b.reset()

        nonstat_sequence_length = 100_000
        low_inds = np.arange(0, nonstat_sequence_length / 10, nonstat_sequence_length / 100, dtype=int)
        med_inds = np.arange(nonstat_sequence_length / 10, nonstat_sequence_length / 2, nonstat_sequence_length / 20,
                             dtype=int)
        high_inds = np.arange(nonstat_sequence_length / 2, nonstat_sequence_length + 1, nonstat_sequence_length / 4,
                              dtype=int)

        state_space = env_a.observation_space
        action_space = env_a.action_space

        ns_dist = MWNSDistribution(seed=0,
                                   state_space=state_space,
                                   action_space=action_space,
                                   current_task=env_a.task_name)
        ns_dist.task_dist.set_prob_task_change(probability=0.0)
        ns_dist.maint_dist.set_prob_maintenance(probability=0.0)
        ns_dist.sensor_dist.set_degradation_params(bias_prob=5e-2)
        ns_dist.actuator_dist.set_degradation_params(bias_prob=5e-2)
        ns_dist.generate_sequence(sequence_length=nonstat_sequence_length + 1)
        ns_dist.freeze()
        env_b.ns_dist = ns_dist

        sampler = MDPDifferenceSampler(environment_a=env_a,
                                       environment_b=env_b,
                                       state_space=state_space,
                                       action_space=action_space)

        env_b.ns_dist.set_sequence_ind(ind=ns_seq_ind)

        for rep in range(repeats):
            dist = sampler.get_difference(n_states=15, n_transitions=2)
            pd_row = pd.DataFrame({'task': [task], 'ns_seq_ind': [ns_seq_ind], 'rep': [rep], 'w1': [dist]})
            w1_dists = pd.concat((w1_dists, pd_row))
    return w1_dists


def make_ns_job_infos(tasks, ns_inds, repeats):
    list_of_dicts = []
    for task in tasks:
        for ind in ns_inds:
            dict = {'tasks': [task], 'ns_seq_ind': [ind], 'repeats': [repeats]}
            list_of_dicts.append(dict)
    return list_of_dicts



nonstat_sequence_length = 100_000
# low_inds = np.arange(0, nonstat_sequence_length / 10, nonstat_sequence_length / 100, dtype=int)
# med_inds = np.arange(nonstat_sequence_length / 10, nonstat_sequence_length/2, nonstat_sequence_length / 20, dtype=int)
# high_inds = np.arange(nonstat_sequence_length / 2, nonstat_sequence_length+1, nonstat_sequence_length / 4, dtype=int)
# nonstat_eval_inds = np.concatenate((low_inds, med_inds, high_inds))
# nonstat_eval_inds = [nonstat_eval_inds[0]] + [nonstat_eval_inds[-1]]
nonstat_eval_inds = np.arange(0, nonstat_sequence_length+1, nonstat_sequence_length / 20, dtype=int)
job_info = make_ns_job_infos(tasks=task_pool_10[0:5], ns_inds=nonstat_eval_inds, repeats=10)
len(job_info)

with mp.Pool(10) as pool:
    w1_dists = pd.concat(pool.map(get_w1_dists_ns, job_info))
print('done')

fig, ax = plt.subplots()
for posind, ind in enumerate(nonstat_eval_inds):
    plotdata = w1_dists.query("ns_seq_ind==" + str(ind))
    plt.boxplot(plotdata['w1'], positions=[posind])
ax.set_xticks(ticks=range(0, posind + 1), labels=nonstat_eval_inds)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()
