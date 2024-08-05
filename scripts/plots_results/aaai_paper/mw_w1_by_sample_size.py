import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10
np.set_printoptions(suppress=True)

def get_job_infos(tasks_a, tasks_b, ns_list, nt_list, repeats):
    job_configs = []
    for rep in range(repeats):
        for task_a in tasks_a:
            for task_b in tasks_b:
                for ns in ns_list:
                    for nt in nt_list:
                        job_configs.append({'task_a': [task_a],
                                            'task_b': [task_b],
                                            'ns': [ns],
                                            'nt': [nt]})
    return job_configs


def mp_calculate_distances(job_config):
    task_a = job_config['task_a'][0]
    task_b = job_config['task_b'][0]
    ns = job_config['ns'][0]
    nt = job_config['nt'][0]
    w1_dists = pd.DataFrame()

    env_a = MetaWorldWrapper(task_name=task_a)
    env_a.change_task(task_name=task_a)
    env_a.reset()

    env_b = MetaWorldWrapper(task_name=task_b)
    env_b.change_task(task_name=task_b)
    env_b.reset()

    sampler_reward_shaped = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                                 state_space=env_a.observation_space, action_space=env_a.action_space,
                                                 method='mcce')
    sampler_random = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                          state_space=env_a.observation_space, action_space=env_a.action_space,
                                          method='random')

    w1_rs = sampler_reward_shaped.get_difference(n_states=ns, n_transitions=nt)
    w1_random = sampler_random.get_difference(n_states=ns, n_transitions=nt)
    pd_row = pd.DataFrame({'task_a': [task_a],
                           'task_b': [task_b],
                           'ns': [ns],
                           'nt': [nt],
                           'w1_rs': [w1_rs],
                           'w1_random': [w1_random]})
    w1_dists = pd.concat((w1_dists, pd_row))
    return w1_dists


job_configs = get_job_infos(tasks_a=task_pool_10,
                            tasks_b=task_pool_10,
                            ns_list=[5, 25, 50],
                            nt_list=[1, 5, 15],
                            repeats=5)
print(len(job_configs))


with mp.Pool(14) as pool:
    w1_dists = pd.concat(pool.map(mp_calculate_distances, job_configs))
print('done')

w1_dists.query('ns==50').groupby(['task_a', 'task_b', 'ns', 'nt'])['w1_rs']
w1_dists.query('ns==50').groupby(['task_a', 'task_b', 'ns', 'nt'])['w1_random'].std()

# with open('TMP.pkl', 'wb') as file:
#     pickle.dump(w1_dists, file)


