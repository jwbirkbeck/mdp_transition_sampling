import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
from src.dqn.dqn_agent import DQNAgent

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

wind_probs = []
evals = []
dists = []
for seed in range(5000):
    new_wind_prob = np.random.uniform(size=1)[0]
    wind_probs.append(new_wind_prob)
    new_wind_prob = torch.tensor(new_wind_prob, requires_grad=False, device=device)
    # agent.environment.windy = True
    # agent.environment.wind_prob = new_wind_prob

    agent.environment.seed = seed

    print(f"{seed}, {new_wind_prob}")
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
import pickle

pickle.dump(results_dict, open('results_dict.pkl', 'wb'))

plt.scatter(dists, evals, alpha=0.5, s=3)
plt.show()

plt.scatter(wind_probs, dists, alpha=0.5, s=3)
plt.show()

plt.hist(dists, bins=100)
plt.xticks(ticks=np.arange(0, 12, 0.5), rotation=-45, ha='left', rotation_mode='anchor')
plt.show()

plt.scatter(wind_probs, dists, alpha=0.5, s=3)
plt.show()

plotdata = pd.DataFrame({'dists': dists, 'evals': evals})

# # bins = [-0.01, 0.5, 1.2, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
# _, bins = pd.qcut(plotdata.dists, q=10, retbins=True)
# fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
# # plt.plot(x[order], y_sm[order], color='black', label='LOWESS', alpha=0.2)
# # plt.fill_between(x[order], y_sm[order] - 1.96*y_std[order], y_sm[order] + 1.96*y_std[order], alpha=0.15, label='LOWESS uncertainty')
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.dists > bin_low, plotdata.dists <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.evals, positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=2e-2)
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SimpleGrid: SOPR vs W1 MDP distance")
plt.xticks(ticks = np.arange(0, 11.5, 1))
plt.tight_layout()
plt.show()


def get_optimal_action(env, minimize=False):
    agent_pos = env._agent_pos
    goal_pos = env._goal_pos
    opt_actions = []
    if not minimize:
        if agent_pos[0] < goal_pos[0]:
            opt_actions.append(1)
        if agent_pos[0] > goal_pos[0]:
            opt_actions.append(3)
        if agent_pos[1] < goal_pos[1]:
            opt_actions.append(2)
        if agent_pos[1] > goal_pos[1]:
            opt_actions.append(0)
        if torch.all(agent_pos == goal_pos):
            # check if we're near an edge. if we are, move into that edge:
            if agent_pos[0] == 1:
                opt_actions.append(3)
            if agent_pos[0] == env.size-2:
                opt_actions.append(1)
            if agent_pos[1] == env.size-2:
                opt_actions.append(2)
    else:
        if agent_pos[0] > goal_pos[0]:
            opt_actions.append(1)
        if agent_pos[0] < goal_pos[0]:
            opt_actions.append(3)
        if agent_pos[1] > goal_pos[1]:
            opt_actions.append(2)
        if agent_pos[1] < goal_pos[1]:
            opt_actions.append(0)
    if len(opt_actions) == 0:
        opt_actions = [0, 1, 2, 3]
    return np.random.choice(opt_actions)


def get_optimal_return(env, minimize=False):
    ep_reward = 0
    env.reset()
    truncated = terminated = False
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(get_optimal_action(env, minimize=minimize))
        ep_reward += reward.item()
        optimal_return = ep_reward
    return optimal_return


def get_max_min_return(env):
    min_return = get_optimal_return(env=env, minimize=True)
    max_return = get_optimal_return(env=env, minimize=False)
    return min_return, max_return


min_rs = []
max_rs = []
for seed in range(5000):
    print(seed)
    env.seed = seed
    min_r, max_r = get_max_min_return(env=env)
    min_rs.append(min_r)
    max_rs.append(max_r)

# plotdata[plotdata.evals > plotdata.max_r].iloc[0]
#
# env.seed = 104
# env.reset()
# env.render()
# truncated = terminated = False
# ep_reward = 0
# while not (truncated or terminated):
#     observation, reward, terminated, truncated, info = env.step(get_optimal_action(env, minimize=False))
#     env.render()
#     ep_reward += reward.item()
#     optimal_return = ep_reward
#
# agent.epsilon = 0.0
# agent.train_agent(train=False, render=True)
# agent.epsilon = 0.1

soprs = [(max_r - r) / (max_r - min_r) for r, max_r, min_r in zip(evals, max_rs, min_rs)]
plotdata = pd.DataFrame({'dists': dists, 'evals': evals, 'sopr': soprs, 'min_r': min_rs, 'max_r': max_rs})
plotdata.min_r[plotdata.evals < plotdata.min_r] = plotdata.evals[plotdata.evals < plotdata.min_r]
plotdata.max_r[plotdata.evals > plotdata.max_r] = plotdata.evals[plotdata.evals > plotdata.max_r]
plotdata.sopr = (plotdata.max_r - plotdata.evals) / (plotdata.max_r - plotdata.min_r)
_, bins = pd.qcut(plotdata.dists, q=16, retbins=True)



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
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SimpleGrid: SOPR vs W1 MDP distance")
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.tight_layout()
plt.savefig("simplegrid_sopr.png", dpi=300)
plt.show()

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
plt.scatter(plotdata.dists, plotdata.sopr, s=3, alpha = 0.5)
plt.show()

# """
# Pre-calculate optimal and minimal returns:
# Optimal returns: Sample actions which step toward goal and then get reward.
# Minimal returns: Head toward furthest edge corner
# """
