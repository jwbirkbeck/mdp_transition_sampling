import torch
import matplotlib.pyplot as plt

from src.dyna_like.SAC_dyna import SACDynaLike
from src.dyna_like.SAC_dyna_random_batch import SACDynaRandom
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper

env = MetaWorldWrapper(task_name='reach-v2')
env.change_task(task_name='reach-v2', task_number=0)
device = torch.device("cpu")
sac_agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=10, memory_length=10, device=device, polyak=0.995)
sac_dyna_agent = SACDynaLike(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                             batch_size=10, memory_length=10, device=device, polyak=0.995)
sac_dyna_random_agent = SACDynaRandom(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                                      batch_size=10, memory_length=10, device=device, polyak=0.995)

range_ind = 250
# sac_rewards = []
# sac_returns = []
# for _ in range(range_ind):
#     rew, ret = sac_agent.train_agent()
#     sac_rewards.append(rew)
#     sac_returns.append(ret)
# print("done")
sac_rewards = []
sac_returns = []
sac_dyna_rewards = []
sac_dyna_returns = []
for _ in range(range_ind):
    rew, ret = sac_agent.train_agent()
    sac_rewards.append(rew)
    sac_returns.append(ret)

    rew, ret = sac_dyna_agent.train_agent()
    sac_dyna_rewards.append(rew)
    sac_dyna_returns.append(ret)
print("done")

sac_dyna_agent.updates
sac_dyna_agent.sim_updates

plt.plot(sac_rewards, label='SAC')
plt.plot(sac_dyna_rewards, label = 'Dynalike SAC')
plt.xlabel("Episode")
plt.ylabel("rewards")
plt.title("SAC vs Wasserstein Dynalike SAC, no replay buffers")
plt.legend()
# plt.savefig("dynalike.png")
plt.show()


sac_dyna_random_rewards = []
for _ in range(range_ind):
    rew, _ = sac_dyna_random_agent.train_agent()
    sac_dyna_random_rewards.append(rew)
print("done")