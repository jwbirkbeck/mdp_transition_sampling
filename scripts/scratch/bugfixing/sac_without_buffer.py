import torch
import numpy as np
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.soft_actor_critic.sac_agent import SACAgent

task_selection = ['handle-press-side-v2',
                  'handle-press-v2',
                  'plate-slide-back-v2',
                  'reach-v2',
                  'reach-wall-v2']

device = torch.device("cuda")
env = MetaWorldWrapper('reach-v2')

agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-3, gamma=0.99,
                 batch_size=1, memory_length=1, device=device, polyak=0.995)

rewards = []
for _ in range(250):
    print(_)
    ep_rew, ep_ret = agent.train_agent()
    rewards.append(ep_rew)
print("done")

plt.plot(rewards)
plt.show()

plt.plot(np.convolve(rewards, np.ones(25)/25, mode='valid'))
plt.show()

rewards[-1]