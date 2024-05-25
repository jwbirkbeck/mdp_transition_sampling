import sys
import os
import torch
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.dyna_like.SAC_dyna import SACDynaLike
from src.metaworld.wrapper import MetaWorldWrapper

results_path = os.path.expanduser('~/mdp_transition_sampling/results/sac_dynalike/')
# results_path = '/opt/project/results/sac_dynalike/'
prev_umask = os.umask(000)
os.makedirs(results_path) if not os.path.isdir(results_path) else None
os.umask(prev_umask)

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

n_eps = 4000
device = torch.device("cpu")
env = MetaWorldWrapper(task_name='reach-v2')

sac_agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=10, memory_length=10, device=device, polyak=0.995)
sac_dynalike = SACDynaLike(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                             batch_size=10, memory_length=10, device=device, polyak=0.995)

results = pd.DataFrame()
for ep in range(n_eps):
    sac_rew, _ = sac_agent.train_agent()
    sac_dynalike_rew, _ = sac_dynalike.train_agent()
    pd_row = pd.DataFrame({'episode': [ep],
                           'sac': [sac_rew],
                           'sac_dynalike': [sac_dynalike_rew]})
    results = pd.concat((results, pd_row))
    results.to_csv(os.path.join(results_path, 'results_' + str(run_index) + ".csv"))
