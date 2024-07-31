import sys
import os
import torch
import pandas as pd
from src.soft_actor_critic.sac_agent_v2 import SACAgentV2
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10, ns_test_inds
from src.utils.funcs import get_standard_ns_dist
from src.utils.filepaths import *

results_dir = os.path.join(results_path_iridis, 'mw_train_agents')
model_dir = os.path.join(model_path_iridis, 'mw_train_agents')

assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")
n_training_eps = 750
n_eps_per_checkpoint = 25
n_nonstat_eval_eps = 10

envs = []
agents = []
for ind, task in enumerate(task_pool_10):
    envs.append(MetaWorldWrapper(task_name=task))
    agents.append(SACAgentV2(environment=envs[ind], hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                             batch_size=500, memory_length=1e6, device=device, polyak=0.995))

train_results = pd.DataFrame()
eval_results = pd.DataFrame()
for task in task_pool_10:
    env = MetaWorldWrapper(task_name=task)
    env.change_task(task_name=task, task_number=0)  # zero variation between episodes to control all nonstationarity
    agent = SACAgentV2(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                       batch_size=500, memory_length=1e6, device=device, polyak=0.995)

    # Train agent in base task without non-stationarity
    for train_rep in range(n_training_eps):
        ep_rew, ep_ret = agent.train_agent()
        pd_row = pd.DataFrame({'task': [task],
                               'episode': [train_rep],
                               'rewards': [ep_rew]})
        train_results = pd.concat((train_results, pd_row))
        if train_rep % n_eps_per_checkpoint == 0:
            train_results.to_csv(os.path.join(results_dir, 'train_' + str(run_index) + '.csv'), index=False)
            agent.save(save_path=os.path.join(model_dir, f'{task}_model_{run_index}'))

    ns_dist = get_standard_ns_dist(env=env)
    ns_dist.freeze()
    env.ns_dist = ns_dist

    for rep in range(n_nonstat_eval_eps):
        for eval_ind in ns_test_inds:
            env.ns_dist.set_sequence_ind(ind=eval_ind)
            ep_rew, _ = agent.train_agent(train=False)
            pd_row = pd.DataFrame({'task': [task],
                                   'rep': [rep],
                                   'test_ind': [eval_ind],
                                   'ep_reward': [ep_rew]})
            eval_results = pd.concat((eval_results, pd_row))
        eval_results.to_csv(os.path.join(results_dir, 'eval_' + str(run_index) + '.csv'), index=False)
