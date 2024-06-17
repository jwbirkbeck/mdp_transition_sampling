import torch
import matplotlib.pyplot as plt
from src.finite_mdps.simple_grid_v1 import SimpleGridV1
from src.dqn.dqn_agent import DQNAgent

device = torch.device('cpu')
env = SimpleGridV1(height=12, width=12, seed=1, device=device, render_mode='human')

agent = DQNAgent(environment=env,
                 alpha=1e-3,
                 epsilon=0.05,
                 gamma=0.99,
                 hidden_layer_sizes=(128, 256),
                 batch_size=500,
                 memory_length=int(1e6),
                 device=device)

rewards = []
for _ in range(50):
    agent.environment.seed = _ % 10
    print(_)
    ep_reward, _ = agent.train_agent()
    rewards.append(ep_reward)
print("done")
plt.plot(rewards)
plt.show()

for _ in range(25):
    agent.environment.seed = _
    agent.train_agent(train=False, render=True)
