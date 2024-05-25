import sys
import os
import torch
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper

task_selection = ['handle-press-side-v2',
                  'handle-press-v2',
                  'plate-slide-back-v2',
                  'reach-v2',
                  'reach-wall-v2']

results_path = os.path.expanduser('~/mdp_transition_sampling/results/simple_w1_vs_returns_test/')
# results_path = '/opt/project/results/simple_w1_vs_returns_test/'
prev_umask = os.umask(000)
os.makedirs(results_path) if not os.path.isdir(results_path) else None
os.umask(prev_umask)

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")
envs = {}
agents = {}
for base_task in task_selection:
    envs[base_task] = MetaWorldWrapper(task_name=base_task)
    agents[base_task] = SACAgent(environment=envs[base_task], hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                           batch_size=500, memory_length=1e6, device=device, polyak=0.995)

n_training_eps = 1000
n_eps_before_eval = 25
n_eval_reps = 10

training = pd.DataFrame()
testing = pd.DataFrame()

training_eps = 0
for _ in range(int(n_training_eps / n_eps_before_eval)):
    for _ in range(n_eps_before_eval):
        for base_task in task_selection:
            agents[base_task].environment.change_task(task_name=base_task)
            rew, ret = agents[base_task].train_agent()
            tr_pd_row = pd.DataFrame({'env': [base_task], 'rewards': [rew], 'episodes': [training_eps]})
            training = pd.concat((training, tr_pd_row))
        training_eps +=1
    training.to_csv(os.path.join(results_path, 'train_' + str(run_index) + '.csv'))

    for base_task in task_selection:
        for comp_task in task_selection:
            agents[base_task].environment.change_task(task_name=comp_task)
            for _ in range(n_eval_reps):
                rew, ret = agents[base_task].train_agent(train=False)
                test_pd_row = pd.DataFrame({'env_a': [base_task], 'env_b': [comp_task], 'rewards': [rew], 'episodes': [training_eps]})
                testing = pd.concat((testing, test_pd_row))
    testing.to_csv(os.path.join(results_path, 'test_' + str(run_index) + '.csv'))