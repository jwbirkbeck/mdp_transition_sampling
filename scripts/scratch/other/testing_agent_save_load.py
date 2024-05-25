"""
Train agent
save agent
load agent
keep training agent - view training performance and ensure smooth

test memory wipe
"""

import torch
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper
import matplotlib.pyplot as plt

env = MetaWorldWrapper(task_name='reach-v2')
device = torch.device('cpu')

agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                 batch_size=500, memory_length=1e6, device=device, polyak=0.995)

rewards = []
for _ in range(25):
    rew, _ = agent.train_agent()
    rewards.append(rew)
print("done")

plt.plot(rewards)
plt.show()

# test that a new dir path is created and can be loaded from
# test that loading from a fake path fails
# test that loading an agent with a different config fails

save_path = '/opt/project/other/agent_save'
agent.save(save_path=save_path)

new_agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=500, memory_length=1e6, device=device, polyak=0.995)

new_agent.load(load_path=save_path)


rewards_new = []
for _ in range(25):
    rew, _ = agent.train_agent()
    rewards_new.append(rew)
print("done")

plt.plot(rewards + rewards_new)
plt.show()

rewards_new2 = []
for _ in range(25):
    rew, _ = new_agent.train_agent()
    rewards_new2.append(rew)
print("done")

plt.plot(rewards + rewards_new2)
plt.show()

# If above works, test that memory wipe works correctly:
new_agent2 = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                      batch_size=500, memory_length=1e6, device=device, polyak=0.995)

new_agent2.load(load_path=save_path)
