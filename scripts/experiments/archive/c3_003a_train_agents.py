import sys
import os
import torch
import numpy as np
import pandas as pd
from time import time
from src.soft_actor_critic.sac_agent_v2 import SACAgentV2
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10
from src.utils.filepaths import *


results_dir = os.path.join(results_path_iridis, 'c3_003a_train_agents')
model_dir = os.path.join(model_path_iridis, 'c3_003a_train_agents')

assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")
n_training_eps = 5000
n_eps_per_checkpoint = 25
n_eval_eps = 5

envs = []
agents = []
for ind, task in enumerate(task_pool_10):
    envs.append(MetaWorldWrapper(task_name=task))
    agents.append(SACAgentV2(environment=envs[ind], hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                             batch_size=500, memory_length=1e6, device=device, polyak=0.995))

train_results = pd.DataFrame()
eval_results = pd.DataFrame()
for rep in range(n_training_eps):
    for ind, task in enumerate(task_pool_10):
        start_time = time()
        ep_rew, _ = agents[ind].train_agent()
        end_time = time()
        pd_row = pd.DataFrame({'task': [task], 'episode': [rep], 'rewards': [ep_rew], 'time': [end_time - start_time]})
        train_results = pd.concat((train_results, pd_row))
        train_results.to_csv(os.path.join(results_dir, f'train_{run_index}.csv'), index=False)
        if rep % n_eps_per_checkpoint == 0:
            # Eval agent n_eval_eps times
            for eval_rep in range(n_eval_eps):
                start_time = time()
                ep_rew, _ = agents[ind].train_agent(train=False)
                end_time = time()
                pd_row = pd.DataFrame({'task': [task], 'episode': [rep], 'rewards': [ep_rew], 'time': [end_time - start_time]})
                eval_results = pd.concat((eval_results, pd_row))
            eval_results.to_csv(os.path.join(results_dir, f'eval_{run_index}.csv'), index=False)
            agents[ind].save(save_path=os.path.join(model_dir, f'{task}_model_{run_index}'))
