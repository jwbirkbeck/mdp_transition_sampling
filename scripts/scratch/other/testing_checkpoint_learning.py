import sys
import os
import torch
import numpy as np
import pandas as pd
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10

device = torch.device('cpu')

env = MetaWorldWrapper(task_name=task_pool_10[0])
agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                 batch_size=500, memory_length=1e6, device=device, polyak=0.995)

n_training_eps = 500
n_eps_checkpointing = 25

rewards = []
prev_running_avg = 0
running_avg = 0
for train_ep in range(n_training_eps):
    reward, _ = agent.train_agent()
    rewards.append(reward)
    running_avg = (0.99 * running_avg) + (0.01 * reward)
    if train_ep % n_eps_checkpointing == 0:
        if running_avg < prev_running_avg:
            agent.load(load_path='/opt/project/models/testing')
        else:
            agent.save(save_path='/opt/project/models/testing')
            prev_running_avg = running_avg
        checkpoint_rewards = []
print('done')

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()