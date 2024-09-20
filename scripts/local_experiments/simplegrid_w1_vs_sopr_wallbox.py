import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.dqn.dqn_agent import DQNAgent
import pickle


device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

evals = []
dists = []
for seed in range(50):
    print(seed)
    agent = DQNAgent(environment=env,
                     alpha=1e-3,
                     epsilon=0.1,
                     gamma=0.9,
                     hidden_layer_sizes=(128, 128),
                     batch_size=100,
                     memory_length=int(1e6),
                     device=device)

    agent.environment.seed = seed

    rewards = []
    for _ in range(25):
        ep_reward, _ = agent.train_agent(train=True)
        rewards.append(ep_reward)
    agent.epsilon = 0.05
    for _ in range(25):
        ep_reward, _ = agent.train_agent(train=True)
        rewards.append(ep_reward)
    agent.epsilon = 0.0
    for _ in range(100):
        ep_reward, _ = agent.train_agent(train=True)
        rewards.append(ep_reward)

    agent.environment.walls_around_agent = True
    ep_reward, _ = agent.train_agent(train=False)
    evals.append(ep_reward)

    env.reset()
    next_states_a, rewards_a = env.get_all_transitions(render=False)
    env.walls_around_agent = True
    env.reset()
    next_states_b, rewards_b = env.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists.append(ot.emd2(a=a, b=b, M=M))
    env.reset()
print('done')

# plt.scatter(dists, evals)
# plt.show()

simplegrid_w1_vs_sopr_wallbox = {'seed': list(range(50)), 'dists': dists, 'evals': evals}
with open('simplegrid_w1_vs_sopr_wallbox.pkl', 'wb') as file:
    pickle.dump(simplegrid_w1_vs_sopr_wallbox, file)
