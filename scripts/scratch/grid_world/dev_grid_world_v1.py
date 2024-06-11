import torch
import matplotlib.pyplot as plt
from src.finite_mdps.grid_world_v1 import GridWorldV1
from src.dqn.dqn_agent import DQNAgent
import cProfile

device = torch.device('cpu')
env = GridWorldV1(height=25, width=25, seed=0, device=device, render_mode='human')

agent = DQNAgent(environment=env,
                 alpha=1e-3,
                 epsilon=0.1,
                 gamma=0.9,
                 hidden_layer_sizes=(128, 128),
                 batch_size=100,
                 memory_length=int(1e6),
                 device=device)

rewards = []

def tester():
    for _ in range(250):
        env.seed = _
        print(_)
        ep_reward, _ = agent.train_agent()
        rewards.append(ep_reward)
    print('done')

cProfile.run("tester()")



# import pstats
# p = pstats.Stats('profiled.stat')
# p.strip_dirs().sort_stats(-1).print_stats()

# plt.plot(rewards)
# plt.show()
#
# agent.environment.reset()
# agent.environment.render()
#
# env.seed = torch.randint(low=0, high=250, size=(1,)).item()
# agent.train_agent(train=True, render=True)
# print("done")
# agent.environment.close()

