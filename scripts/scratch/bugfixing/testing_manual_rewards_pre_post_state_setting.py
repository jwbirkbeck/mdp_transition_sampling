import numpy as np
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10

"""
For each task in pool:

sample a random trajectory through environment
record states, actions, rewards
calculate manual rewards

set the state of the MDP, execute, get rewards
compare

"""

env = MetaWorldWrapper(task_name='handle-press-v2')
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
    calcd_rewards.append(calcd_reward[0])

plt.hist(np.array(calcd_rewards) - np.array(true_rewards))
plt.show()

env.reset()
manual_next_states = []
manual_rewards = []
manual_calcd_rewards = []
for state, action in zip(states, actions):
    next_state, reward, done, truncated, info = env.get_manual_step(state=state, action=action)
    manual_next_states.append(next_state)
    manual_rewards.append(reward)
    man_calc_reward = env.compute_reward_wrap(states=[state], actions=[action])
    manual_calcd_rewards.append(man_calc_reward)
