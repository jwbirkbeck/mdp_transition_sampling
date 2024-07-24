import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
from src.dqn.dqn_agent import DQNAgent
import pickle


device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')


agent = DQNAgent(environment=env,
                 alpha=1e-3,
                 epsilon=0.1,
                 gamma=0.9,
                 hidden_layer_sizes=(128, 128),
                 batch_size=100,
                 memory_length=int(1e6),
                 device=device)

rewards = []
for _ in range(25):
    agent.environment.seed = 0
    print(_)
    ep_reward, _ = agent.train_agent(train=True)
    rewards.append(ep_reward)
agent.epsilon = 0.05
for _ in range(100):
    agent.environment.seed = 0
    print(_)
    ep_reward, _ = agent.train_agent(train=True)
    rewards.append(ep_reward)
agent.epsilon = 0.0
for _ in range(25):
    agent.environment.seed = 0
    print(_)
    ep_reward, _ = agent.train_agent(train=True)
    rewards.append(ep_reward)
print('done')
plt.plot(rewards)
plt.show()

env_0 = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

evals = []
dists = []
for seed in range(5000):
    print(seed)
    agent.environment.seed = seed

    ep_reward, _ = agent.train_agent(train=False)
    evals.append(ep_reward)

    env.reset()
    env_0.reset()

    next_states_a, rewards_a = env.get_all_transitions(render=False)
    next_states_b, rewards_b = env_0.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists.append(ot.emd2(a=a, b=b, M=M))
    env.reset()
    env_0.reset()
print('done')

results_dict = {'dists': dists,
                'evals': evals}

pickle.dump(results_dict, open('simplegrid_w1_vs_returns.pkl', 'wb'))