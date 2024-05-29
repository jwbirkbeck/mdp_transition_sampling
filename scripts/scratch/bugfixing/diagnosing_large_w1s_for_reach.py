import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metaworld.nonstationarity_distribution import MWNSDistribution
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10
np.set_printoptions(suppress=True)

task = 'reach-v2'
env_a = MetaWorldWrapper(task_name=task)
env_b = MetaWorldWrapper(task_name=task)
state_space = env_a.observation_space
action_space = env_a.action_space

env_a.reset()
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

ns_dist.set_sequence_ind(ind=50000)

sampler = MDPDifferenceSampler(environment_a=env_a,
                               environment_b=env_b,
                               state_space=state_space,
                               action_space=action_space)

for rep in range(3):
    dist = sampler.get_difference(n_states=15, n_transitions=2)
    print(dist)

env_b.ns_dist