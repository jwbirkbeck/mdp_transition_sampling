import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10, ns_test_inds
from src.utils.funcs import get_standard_ns_dist
np.set_printoptions(suppress=True)

for task in ['door-unlock-v2']:
    env_a = MetaWorldWrapper(task_name=task)
    env_a.change_task(task_name=task)
    env_a.reset()

    env_b = MetaWorldWrapper(task_name=task)
    env_b.change_task(task_name=task)
    env_b.reset()

    ns_dist = get_standard_ns_dist(env=env_a)  # env_a == env_b
    ns_dist.freeze()
    env_b.ns_dist = ns_dist

    env_b.ns_dist.set_sequence_ind(ind=0)

    sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                   state_space=env_a.observation_space, action_space=env_a.action_space,
                                   method='mcce')

    dist = sampler.get_difference(n_states=50, n_transitions=1)
