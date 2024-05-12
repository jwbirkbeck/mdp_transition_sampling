import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.policy_selector.worldmodel_policy_selector import WorldModelPolicySelector
from src.metaworld.wrapper import MetaWorldWrapper
from src.soft_actor_critic.sac_agent import SACAgent

task_selection = ['handle-press-side-v2', 'handle-press-v2', 'plate-slide-back-v2', 'reach-v2', 'reach-wall-v2']
env = MetaWorldWrapper(task_name=task_selection[0])

device = torch.device('cpu')
n_training_eps = 1000

sac_agent = SACAgent(environment=env,
                     hidden_layer_sizes=(128, 256),
                     alpha=1e-4,
                     gamma=0.99,
                     batch_size=1,
                     memory_length=500,
                     device=device,
                     polyak=0.995)

sac_rewards = []
for _ in range(n_training_eps):
    print(_)
    ep_rew, ep_ret = sac_agent.train_agent()
    sac_rewards.append(ep_rew)
print("done")

plt.plot(sac_rewards)
plt.show()

policy_selector = WorldModelPolicySelector(env=env, n_policies=len(task_selection), device=device)
rewards = None
active_inds = None
w1_dists = None
n_pretrain_eps = 10
for ind, task in enumerate(task_selection):
    print(task)
    policy_selector.env.change_task(task_name=task, task_number=0)
    for i in range(n_pretrain_eps):
        print(i)
        policy_selector.active_ind = ind
        ep_rewards, _, active_ind, ep_w1_dists = policy_selector.execute_episode(method='episodic')
        policy_selector.active_ind = ind
        active_ind = ind
        if rewards is None:
            rewards = np.array([[ep_rewards]])
            active_inds = np.array([[active_ind]])
            w1_dists = np.array([ep_w1_dists])
        else:
            rewards = np.concatenate((rewards, np.array([[ep_rewards]])))
            active_inds = np.concatenate((active_inds, np.array([[active_ind]])))
            w1_dists = np.concatenate((w1_dists, np.array([ep_w1_dists])))
print("done")

plt.plot(w1_dists)
plt.show()

plt.plot(rewards)
plt.show()

plt.plot(active_inds)
plt.show()

rewards = active_inds = w1_dists = None
for ep in range(100):
    print(ep)
    ep_rewards, _, ep_active_inds, ep_w1_dists = policy_selector.execute_episode(method='decaying average')
    if rewards is None:
        rewards = np.array([[ep_rewards]])
        active_inds = ep_active_inds
        w1_dists = ep_w1_dists
    else:
        rewards = np.concatenate((rewards, np.array([[ep_rewards]])))
        active_inds = np.concatenate((active_inds, ep_active_inds))
        w1_dists = np.concatenate((w1_dists, ep_w1_dists), axis=0)
    if np.random.uniform() < 0.00:
        new_tasks = [i for i in task_selection if i != policy_selector.env.task_name]
        policy_selector.env.change_task(task_name=new_tasks[np.random.randint(0, 4)])
print("done")

policy_selector.env.task
policy_selector.env.change_task(task_name=task_selection[2], task_number=0)

plt.plot(w1_dists)
plt.show()

plt.plot(rewards)
plt.show()

plt.plot(active_inds)
plt.show()



"""
How to test policy selection:
    * Performance vs SAC in single task scenario
    * Performance vs SAC in task swapping scenarios
    * Performance vs SAC in continuous scenarios
    * Performance vs SAC in semi-continuous scenarios
"""
