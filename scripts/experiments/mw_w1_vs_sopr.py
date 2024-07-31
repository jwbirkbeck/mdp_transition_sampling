import os, sys, glob
import torch
import numpy as np
import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.soft_actor_critic.sac_agent_v2 import SACAgentV2
from src.utils.consts import task_pool_10, ns_test_inds
from src.utils.funcs import get_standard_ns_dist
from src.utils.filepaths import *

results_dir = os.path.join(results_path_iridis, 'mw_w1_vs_sopr')
model_dir = os.path.join(model_path_iridis, 'c3_003a_train_agents')

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

# load the saved agents for each task
# for each test ind in local_experiments/mw_w1_dists, evaluate the agent n=10 times

n_evals = 10
device = torch.device('cpu')

env = MetaWorldWrapper(task_name=task_pool_10[0])
env.change_task(task_name=task_pool_10[0])
env.reset()

ns_dist = get_standard_ns_dist(env=env)
ns_dist.freeze()
env.ns_dist = ns_dist

agents = []
for task_ind, task in enumerate(task_pool_10):
    agents.append(SACAgentV2(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                             batch_size=500, memory_length=1e6, device=device, polyak=0.995))
    agents[task_ind].load(load_path=os.path.join(model_dir, f'{task}_model_{run_index}'))

results = pd.DataFrame()
for eval_ind in range(n_evals):
    for task_ind, task in enumerate(task_pool_10):
        env.change_task(task_name=task)
        env.reset()
        for test_ind in ns_test_inds:
            env.ns_dist.set_sequence_ind(ind=test_ind)
            ep_rew, _ = agents[task_ind].train_agent(train=False)
            print(ep_rew)
            pd_row = pd.DataFrame({'task': [task], 'test_ind': [test_ind],'reward': [ep_rew], 'eval_ind': [eval_ind]})
            results = pd.concat((results, pd_row))
            results.to_csv(os.path.join(results_dir, f'eval_{run_index}.csv'), index=False)