import glob, sys, os
import torch
import kmedoids
import pickle
import numpy as np
import pandas as pd
from src.policy_selector.simple_policy_selector import SimplePolicySelector
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10

results_dir = "~/mdp_transition_sampling/results/"
# results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'medoid_policy_selection/')

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device('cpu')
n_episodes = 6000
p_task_change = 0.05

env = MetaWorldWrapper(task_name=task_pool_10[0])

mapping_save_dir = '/home/jwgb1n21/mdp_transition_sampling/src/policy_selector/'
with open(os.path.join(mapping_save_dir, 'mapping_3.pickle'), 'rb') as file:
    mapping_3 = pickle.load(file)
with open(os.path.join(mapping_save_dir, 'mapping_6.pickle'), 'rb') as file:
    mapping_6 = pickle.load(file)
with open(os.path.join(mapping_save_dir, 'mapping_10.pickle'), 'rb') as file:
    mapping_10 = pickle.load(file)
with open(os.path.join(mapping_save_dir, 'mapping_3r.pickle'), 'rb') as file:
    mapping_3r = pickle.load(file)
with open(os.path.join(mapping_save_dir, 'mapping_6r.pickle'), 'rb') as file:
    mapping_6r = pickle.load(file)

sac_agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)
policy_selector_3 = SimplePolicySelector(env=env, method='precomputed', device=device, task_policy_mapping=mapping_3)
policy_selector_6 = SimplePolicySelector(env=env, method='precomputed', device=device, task_policy_mapping=mapping_6)
policy_selector_10 = SimplePolicySelector(env=env, method='precomputed', device=device, task_policy_mapping=mapping_10)
policy_selector_3r = SimplePolicySelector(env=env, method='precomputed', device=device, task_policy_mapping=mapping_3r)
policy_selector_6r = SimplePolicySelector(env=env, method='precomputed', device=device, task_policy_mapping=mapping_6r)

results = pd.DataFrame()
for n_ep in range(n_episodes):
    if np.random.uniform() < p_task_change:
        new_task = np.random.choice(task_pool_10)
        env.change_task(task_name=new_task)
    sac_ep_ret, _ = sac_agent.train_agent()
    ps3_ep_ret, _ = policy_selector_3.play_episode()
    ps6_ep_ret, _ = policy_selector_6.play_episode()
    ps10_ep_ret, _ = policy_selector_10.play_episode()
    ps3r_ep_ret, _ = policy_selector_3r.play_episode()
    ps6r_ep_ret, _ = policy_selector_6r.play_episode()

    pd_row = pd.DataFrame({'episode': [n_ep],
                           'sac': [sac_ep_ret],
                           'ps3': [ps3_ep_ret],
                           'ps6': [ps6_ep_ret],
                           'p10': [ps10_ep_ret],
                           'ps3r': [ps3r_ep_ret],
                           'ps6r': [ps6r_ep_ret]})
    results = pd.concat([results, pd_row])
    csv_name = "results_" + str(run_index) + ".csv"
    results.to_csv(os.path.join(results_path, csv_name), index=False)
