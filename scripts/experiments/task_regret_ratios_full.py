import sys
import os
import torch
import numpy as np
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool

results_path = os.path.expanduser('~/mdp_transition_sampling/results/task_regret_ratios_full/')
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
base_index = int(sys.argv[1])

n_eps_total = 4000
n_eps_per_eval = 50
n_eval_reps = 10
base_task = task_pool[base_index]
device = torch.device('cpu')
task_num = 0

opt_env = MetaWorldWrapper(task_name=base_task)
min_env = MetaWorldWrapper(task_name=base_task)

opt_agent = SACAgent(environment=opt_env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)
min_agent = SACAgent(environment=min_env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)

opt_agent.environment.change_task(task_name=base_task, task_number=task_num)
min_agent.environment.change_task(task_name=base_task, task_number=task_num)

train_episodes_iter = 0
train_results = pd.DataFrame()
eval_results = pd.DataFrame()
for _ in range(int(n_eps_total / n_eps_per_eval)):
    for _ in range(n_eps_per_eval):
        # Fix the task back to the selected task variant to reduce uncontrolled results variance:
        opt_agent.environment.change_task(task_name=base_task, task_number=task_num)
        min_agent.environment.change_task(task_name=base_task, task_number=task_num)
        # Train optimal and minimal agents
        min_agent.environment.negate_rewards = True
        opt_rewards_tr, opt_returns_tr = opt_agent.train_agent()
        min_rewards_tr, min_returns_tr = min_agent.train_agent()
        train_episodes_iter += 1
        tr_pd_row = pd.DataFrame({'agent': ['opt_agent', 'min_agent'],
                                  'episodes': [train_episodes_iter, train_episodes_iter],
                                  'rewards': [opt_rewards_tr, min_rewards_tr],
                                  'returns': [opt_returns_tr, min_returns_tr]})
        train_results = pd.concat((train_results, tr_pd_row))
        tr_csv_name = 'train_results_' + str(base_index) + ".csv"
        train_results.to_csv(results_path + tr_csv_name, index=False)

    opt_agent.save(os.path.join(results_path, 'opt_agent_' + base_task + '/'))
    min_agent.save(os.path.join(results_path, 'min_agent_' + base_task + '/'))

    # Evaluate multiple times:
    for _ in range(n_eval_reps):
        # Evaluate minimal agent:
        min_agent.environment.change_task(task_name=base_task, task_number=task_num)
        min_agent.environment.negate_rewards = False
        min_rewards_eval, min_returns_eval = min_agent.train_agent(train=False)

        eval_pd_row = pd.DataFrame({'agent': ['min_agent'],
                                    'train_episodes': [train_episodes_iter],
                                    'env_a': [base_task],
                                    'env_b': [base_task],
                                    'rewards': [min_rewards_eval],
                                    'returns': [min_returns_eval]})
        eval_results = pd.concat((eval_results, eval_pd_row))

        # Evaluate optimal agent in every environment:
        for comp_task in task_pool:
            opt_agent.environment.change_task(task_name=comp_task, task_number=task_num)
            opt_rewards_eval, opt_returns_eval = min_agent.train_agent(train=False)
            eval_pd_row = pd.DataFrame({'agent': ['opt_agent'],
                                        'train_episodes': [train_episodes_iter],
                                        'env_a': [base_task],
                                        'env_b': [comp_task],
                                        'rewards': [opt_rewards_eval],
                                        'returns': [opt_returns_eval]})
            eval_results = pd.concat((eval_results, eval_pd_row))

        eval_csv_name = 'eval_results_' + str(base_index) + ".csv"
        eval_results.to_csv(results_path + eval_csv_name, index=False)
