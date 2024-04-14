import sys
import os
import torch
import numpy as np
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.metaworld.nonstationarity_distribution import MWNSDistribution

task_selection = ['handle-press-side-v2',
                  'handle-press-v2',
                  'plate-slide-back-v2',
                  'reach-v2',
                  'reach-wall-v2']

results_path = os.path.expanduser('~/mdp_transition_sampling/results/cont_nonstat_w1_vs_returns/')
# results_path = '/opt/project/results/cont_nonstat_w1_vs_returns/'
prev_umask = os.umask(000)
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")

n_training_eps = 750
nonstat_sequence_length = 100_000
low_inds = np.arange(0, nonstat_sequence_length / 10, nonstat_sequence_length / 100, dtype=int)
med_inds = np.arange(nonstat_sequence_length / 10, nonstat_sequence_length/2, nonstat_sequence_length / 20, dtype=int)
high_inds = np.arange(nonstat_sequence_length / 2, nonstat_sequence_length+1, nonstat_sequence_length / 4, dtype=int)

nonstat_eval_inds = np.concatenate((low_inds, med_inds, high_inds))
nonstat_eval_reps = 10

train_results = pd.DataFrame()
eval_results = pd.DataFrame()
for task in task_selection:
    env = MetaWorldWrapper(task_name=task)
    env.change_task(task_name=task, task_number=0)  # zero variation between episodes to control all nonstationarity
    agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)

    # Train agent in default task
    for train_rep in range(n_training_eps):
        ep_rew, ep_ret = agent.train_agent()
        pd_row = pd.DataFrame({'task': [task],
                               'episode': [train_rep],
                               'rewards': [ep_rew]})
        train_results = pd.concat((train_results, pd_row))
    train_results.to_csv(os.path.join(results_path, 'train_' + str(run_index) + '.csv'), index=False)

    ns_dist = MWNSDistribution(seed=0,
                               state_space=env.observation_space,
                               action_space=env.action_space,
                               current_task=task)
    ns_dist.task_dist.set_prob_task_change(probability=0.0)
    ns_dist.maint_dist.set_prob_maintenance(probability=0.0)
    ns_dist.generate_sequence(sequence_length=nonstat_sequence_length+1)
    ns_dist.freeze()

    env.ns_dist = ns_dist

    for rep in range(nonstat_eval_reps):
        for eval_ind in nonstat_eval_inds:
            env.ns_dist.set_sequence_ind(ind=eval_ind)
            ep_rew, ep_ret = agent.train_agent(train=False)
            pd_row = pd.DataFrame({'task': [task],
                                   'rep': [rep],
                                   'test_ind': [eval_ind],
                                   'ep_reward': [ep_rew]})
            eval_results = pd.concat((eval_results, pd_row))
    eval_results.to_csv(os.path.join(results_path, 'eval_' + str(run_index) + '.csv'), index=False)