import numpy as np
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool


for task in task_pool:
    env = MetaWorldWrapper(task_name=task)

    env.reset()
    done = truncated = False
    states = []
    actions = []
    true_next_states = []
    true_rewards = []
    calcd_rewards = []
    while not (done or truncated):
        action = env.action_space.sample()
        state = env.get_observation()
        next_state, reward, done, truncated, info = env.step(action)
        states.append(state)
        actions.append(action)
        true_next_states.append(next_state)
        true_rewards.append(reward)

        calcd_reward = env.compute_reward_wrap(states=[state], actions=[action])
        calcd_rewards.append(calcd_reward)

    true_rewards = np.array(true_rewards).reshape(-1)
    calcd_rewards = np.array(calcd_rewards).reshape(-1)

    if all(true_rewards - calcd_rewards < 1e-8):
        print("all rewards match for task " + task)
    else:
        print("reward mismatch for task " + task + ", plots_results")
        plt.hist(true_rewards - calcd_rewards)
        plt.title(task)
        plt.show()
