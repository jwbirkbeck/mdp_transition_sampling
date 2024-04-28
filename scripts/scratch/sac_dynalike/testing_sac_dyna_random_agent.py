import torch
import matplotlib.pyplot as plt
from src.dyna_like.SAC_dyna_random_batch import SACDynaRandom
from src.soft_actor_critic.sac_agent import SACAgent
from src.metaworld.wrapper import MetaWorldWrapper


env = MetaWorldWrapper(task_name='reach-v2')
env.change_task(task_name='reach-v2', task_number=0)
device = torch.device("cpu")
sac_agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                     batch_size=10, memory_length=10, device=device, polyak=0.995)
sac_dyna_random_agent = SACDynaRandom(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                                      batch_size=500, memory_length=1e6, device=device, polyak=0.995)

range_ind = 100
sac_dyna_random_rewards = []
for _ in range(range_ind):
    rew, _ = sac_dyna_random_agent.train_agent()
    sac_dyna_random_rewards.append(rew)
print("done")

print(sac_dyna_random_agent.sim_updates)

plt.plot(sac_dyna_random_rewards)
plt.show()

plt.plot(sac_dyna_random_agent.wm_losses)
plt.show()