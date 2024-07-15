import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.finite_mdps.simple_grid_v1 import SimpleGridV1
from src.dqn.dqn_agent import DQNAgent

device = torch.device('cpu')
env = SimpleGridV1(height=16, width=16, seed=0, device=device, render_mode='human')

agent = DQNAgent(environment=env,
                 alpha=1e-3,
                 epsilon=0.1,
                 gamma=0.9,
                 hidden_layer_sizes=(128, 128),
                 batch_size=100,
                 memory_length=int(1e6),
                 device=device)

rewards = []
for _ in range(100):
    agent.environment.seed = 0
    print(_)
    ep_reward, _ = agent.train_agent(train=True)
    rewards.append(ep_reward)
print('done')

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Episode total reward')
plt.title('DQN performance, SimpleGridV1')
plt.tight_layout()
plt.show()

env_0 = SimpleGridV1(height=16, width=16, seed=0, device=device, render_mode='human')

evals = []
dists = []
for seed in range(1000):
    agent.environment.seed = seed
    print(seed)
    ep_reward, _ = agent.train_agent(train=False)
    evals.append(ep_reward)

    next_states_a, rewards_a = env.get_all_transitions(render=False)
    next_states_b, rewards_b = env_0.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists.append(ot.emd2(a=a, b=b, M=M))
print('done')

plt.scatter(dists, evals, alpha=0.5, s=3)
plt.show()

plotdata = pd.DataFrame({'dists': dists, 'evals': evals})

bins = [-0.01, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

# _, bins = pd.qcut(plotdata.w1, q=16, retbins=True)
fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
# plt.plot(x[order], y_sm[order], color='black', label='LOWESS', alpha=0.2)
# plt.fill_between(x[order], y_sm[order] - 1.96*y_std[order], y_sm[order] + 1.96*y_std[order], alpha=0.15, label='LOWESS uncertainty')
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.dists > bin_low, plotdata.dists <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.evals, positions=[position], showmedians=True, showextrema=False, widths=0.15, bw_method=2e-2)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SOPR against Wasserstein MDP distance")
plt.xticks(ticks = np.arange(0, 2.1, 2/10), rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()
