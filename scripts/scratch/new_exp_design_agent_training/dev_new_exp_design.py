import sys, os, glob, time
import torch
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.filepaths import *
from src.utils.consts import task_pool_10


assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

results_path = os.path.join(results_path_iridis, 'template')
device = torch.device("cpu")
n_training_eps = 2000
n_eps_per_checkpoint = 50

task = task_pool_10[run_index % len(task_pool_10)]
for ep in range(n_training_eps):
    # Train agent
    # Save every n_eps_per_checkpoint
    pass
