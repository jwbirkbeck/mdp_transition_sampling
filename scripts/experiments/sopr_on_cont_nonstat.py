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

results_path = os.path.expanduser('~/mdp_transition_sampling/results/sopr_on_cont_nonstat/')
# results_path = '/opt/project/results/sopr_on_cont_nonstat/'
prev_umask = os.umask(000)
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])
device = torch.device("cpu")

# To spread the starting task across runs, rotate the task_selection list by run_index:
rotator = run_index % len(task_selection)
task_selection = task_selection[rotator:] + task_selection[:rotator]

n_training_eps = 750
n_eval_eps = 20

nonstat_sequence_length = 100_000
low_inds = np.arange(0, nonstat_sequence_length / 10, nonstat_sequence_length / 100, dtype=int)
med_inds = np.arange(nonstat_sequence_length / 10, nonstat_sequence_length/2, nonstat_sequence_length / 20, dtype=int)
high_inds = np.arange(nonstat_sequence_length / 2, nonstat_sequence_length+1, nonstat_sequence_length / 4, dtype=int)
nonstat_eval_inds = np.concatenate((low_inds, med_inds, high_inds))

tr_csv_name = 'train_' + str(run_index) + ".csv"
eval_csv_name = 'eval_' + str(run_index) + ".csv"

train_pd = pd.DataFrame()
eval_pd = pd.DataFrame()
for task in task_selection:
    opt_env = MetaWorldWrapper(task_name=task)
    min_env = MetaWorldWrapper(task_name=task)
    opt_env.change_task(task_name=task, task_number=0)  # zero variation between episodes to control all nonstationarity
    min_env.change_task(task_name=task, task_number=0)  # zero variation between episodes to control all nonstationarity
    opt_agent = SACAgent(environment=opt_env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                         batch_size=500, memory_length=1e6, device=device, polyak=0.995)
    min_agent = SACAgent(environment=min_env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                         batch_size=500, memory_length=1e6, device=device, polyak=0.995)

    min_env.negate_rewards = True

    ns_dist = MWNSDistribution(seed=0,
                               state_space=opt_env.observation_space,
                               action_space=opt_env.action_space,
                               current_task=task)
    ns_dist.task_dist.set_prob_task_change(probability=0.0)
    ns_dist.maint_dist.set_prob_maintenance(probability=0.0)
    ns_dist.generate_sequence(sequence_length=nonstat_sequence_length+1)
    ns_dist.freeze()

    opt_agent.environment.ns_dist = ns_dist
    min_agent.environment.ns_dist = ns_dist

    for raw_ind, test_ind in enumerate(nonstat_eval_inds):
        # Set nonstationarity
        opt_agent.environment.ns_dist.set_sequence_ind(ind=test_ind)
        min_agent.environment.ns_dist.set_sequence_ind(ind=test_ind)

        # Train opt and min agents, save results
        for train_ind in range(n_training_eps):
            rew, ret = opt_agent.train_agent()
            pd_row = pd.DataFrame({'agent': ['opt'],
                                   'task': [task],
                                   'test_ind': [test_ind],
                                   'episode': [train_ind],
                                   'rewards': [rew],
                                   'returns': [ret]})
            train_pd = pd.concat((train_pd, pd_row))
            rew, ret = min_agent.train_agent()
            pd_row = pd.DataFrame({'agent': ['min'],
                                   'task': [task],
                                   'test_ind': [test_ind],
                                   'episode': [train_ind],
                                   'rewards': [rew],
                                   'returns': [ret]})
            train_pd = pd.concat((train_pd, pd_row))
            train_pd.to_csv(results_path + tr_csv_name, index=False)

        # For this loop:
            # Eval opt agent in current MDP
            # Eval min agent in current MDP
        for eval_ind in range(n_eval_eps):
            rew, ret = opt_agent.train_agent(train=False)
            pd_row = pd.DataFrame(
                {'agent': ['opt'],
                 'task': [task],
                 'base_ind':[test_ind],
                 'test_ind': [test_ind],
                 'episode': [eval_ind],
                 'rewards': [rew],
                 'returns': [ret]})
            eval_pd = pd.concat((eval_pd, pd_row))
            eval_pd.to_csv(results_path + eval_csv_name, index=False)

            rew, ret = min_agent.train_agent(train=False)
            pd_row = pd.DataFrame(
                {'agent': ['min'],
                 'task': [task],
                 'base_ind': [test_ind],
                 'test_ind': [test_ind],
                 'episode': [eval_ind],
                 'rewards': [rew],
                 'returns': [ret]})
            eval_pd = pd.concat((eval_pd, pd_row))
            eval_pd.to_csv(results_path + eval_csv_name, index=False)

        # If at the first loop, then evaluate the optimal agent against all future test_inds:
        if raw_ind == 0:
            for future_test_ind in nonstat_eval_inds[1:]:
                opt_agent.environment.ns_dist.set_sequence_ind(ind=future_test_ind)
                for eval_rep in range(n_eval_eps):
                    rew, ret = opt_agent.train_agent(train=False)
                    pd_row = pd.DataFrame(
                        {'agent': ['opt'],
                         'task': [task],
                         'base_ind': [test_ind],
                         'test_ind': [future_test_ind],
                         'episode': [eval_rep],
                         'rewards': [rew],
                         'returns': [ret]})
                    eval_pd = pd.concat((eval_pd, pd_row))
                eval_pd.to_csv(results_path + eval_csv_name, index=False)