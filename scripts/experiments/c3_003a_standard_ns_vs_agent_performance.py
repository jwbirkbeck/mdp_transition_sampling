import sys
import os
import torch
import numpy as np
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10, ns_test_inds
from src.utils.funcs import get_standard_ns_dist

"""
For each of the ten tasks,
    Train an agent until convergence
    Test the agent against each ns in the sequence
"""

results_path = os.path.expanduser('~/mdp_transition_sampling/results/c3_003a_standard_ns_vs_agent_performance/')

assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")
n_training_eps = 1000
n_eps_between_checkpoint = 25
n_eval_reps = 10
train_results = pd.DataFrame()
eval_results = pd.DataFrame()

for task in task_pool_10:
    env = MetaWorldWrapper(task_name=task)
    agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)

    # Train agent in default task with checkpointing
    for train_rep in range(n_training_eps):
        ep_rew, ep_ret = agent.train_agent()
        pd_row = pd.DataFrame({'task': [task],
                               'episode': [train_rep],
                               'rewards': [ep_rew]})
        train_results = pd.concat((train_results, pd_row))
        train_results.to_csv(os.path.join(results_path, f'train_{run_index}.csv'), index=False)

    # Evaluate agent against increasing non-stationarity
    for rep in range(n_eval_reps):
        for test_ind in ns_test_inds:
            env.ns_dist.set_sequence_ind(ind=test_ind)
            ep_rew, ep_ret = agent.train_agent(train=False)
            pd_row = pd.DataFrame({'task': [task],
                                   'rep': [rep],
                                   'test_ind': [test_ind],
                                   'ep_reward': [ep_rew]})
            eval_results = pd.concat((eval_results, pd_row))
            eval_results.to_csv(os.path.join(results_path, f'eval_{run_index}.csv'), index=False)