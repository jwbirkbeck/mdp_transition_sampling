import glob, sys, os
import torch
import pickle
import numpy as np
import pandas as pd
from src.policy_selector.simple_policy_selector import SimplePolicySelector
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10

results_dir = "~/mdp_transition_sampling/results/"
# results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'bandit_policy_selection/')

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device('cpu')
n_episodes = 1500
p_task_change = 0.05

env = MetaWorldWrapper(task_name=task_pool_10[0])

policy_selector_3 = SimplePolicySelector(env=env, method='bandit', device=device, task_names=task_pool_10, n_policies=3)
policy_selector_6 = SimplePolicySelector(env=env, method='bandit', device=device, task_names=task_pool_10, n_policies=6)
policy_selector_10 = SimplePolicySelector(env=env, method='bandit', device=device, task_names=task_pool_10, n_policies=10)
results = pd.DataFrame()
for n_ep in range(n_episodes):
    if np.random.uniform() < p_task_change:
        new_task = np.random.choice(task_pool_10)
        env.change_task(task_name=new_task)
    bps3_ep_ret, _ = policy_selector_3.play_episode()
    bps6_ep_ret, _ = policy_selector_6.play_episode()
    bps10_ep_ret, _ = policy_selector_10.play_episode()

    pd_row = pd.DataFrame({'episode': [n_ep],
                           'bps3': [bps3_ep_ret],
                           'bps6': [bps6_ep_ret],
                           'bps10': [bps10_ep_ret]})
    results = pd.concat([results, pd_row])
    csv_name = "results_" + str(run_index) + ".csv"
    results.to_csv(os.path.join(results_path, csv_name), index=False)
