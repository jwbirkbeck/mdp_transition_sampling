import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.sampler.samplers import MDPDifferenceSampler
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper

from src.utils.consts import task_pool
task_pool = task_pool[0:10]

w1_dists = pd.DataFrame()
for task_a in task_pool:
    for task_b in task_pool:
        # calc w1 dist
        # store in w1_dists
        env_a = MetaWorldWrapper(task_name=task_a)
        env_b = MetaWorldWrapper(task_name=task_b)
        env_a.reset()
        env_b.reset()

        state_bounds = env_a.observation_space
        action_bounds = env_a.action_space
        sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                       state_space=state_bounds, action_space=action_bounds)
        dist = sampler.get_difference(n_states=25, n_transitions=5)
        pd_row = pd.DataFrame({'task_a': [task_a], 'task_b': [task_b], 'w1': [dist]})
        w1_dists = pd.concat([w1_dists, pd_row])
        print(task_b)
print('done')

w1_dists['w1'].plot()
plt.show()

import scipy

"""
# Find triangle of most distant 
Assign P most distant MDPs to each of P policies
For each P policies:
    Select nearest MDP, add it to P's pool 
"""

w1_dists.sort_values('w1')


"""
Select N environments, P policies
Measure the Wasserstein distance between N environments

    Control: Radnomly assign N environments to one of P policies  
    Test: Assign N envs to P policies to minimise the class Wasserstein distance

For control and test, execute the training of the agents over the same sequence of MDP changes while
storing the lifetime rewards
"""
