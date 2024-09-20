import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.dqn.dqn_agent import DQNAgent


"""
Pre-calculate optimal and minimal returns:

Optimal returns: Sample actions which step toward goal and then get reward.  

Minimal returns: Head toward furthest edge corner

"""

device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

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
    min_return = get_optimal_return(minimize=True)
    max_return = get_optimal_return(minimize=False)
    return min_return, max_return

ep_reward = 0
env.reset()
truncated = terminated = False
while not (truncated or terminated):
    observation, reward, terminated, truncated, info = env.step(get_optimal_action(env, minimize=False))
    ep_reward += reward.item()
    max_return = ep_reward

ep_reward = 0
env.reset()
truncated = terminated = False
while not (truncated or terminated):
    observation, reward, terminated, truncated, info = env.step(get_optimal_action(env, minimize=True))
    ep_reward += reward.item()
    min_return = ep_reward

