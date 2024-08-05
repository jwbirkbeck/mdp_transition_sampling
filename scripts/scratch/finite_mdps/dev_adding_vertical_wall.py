import torch
import numpy as np
import ot
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
from src.dqn.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

size = 20
device = torch.device('cpu')
env1 = SimpleGridV2(size=size, agent_pos = torch.tensor([5, 17]), goal_pos = torch.tensor([17, 2]), device=device, render_mode='human')

env1.vertical_wall = True
env1.reset()
env1.render()

agent1 = DQNAgent(environment=env1,
                 alpha=1e-3,
                 epsilon=0.1,
                 gamma=0.9,
                 hidden_layer_sizes=(128, 128),
                 batch_size=100,
                 memory_length=int(1e6),
                 device=device)

rewards1 = []
for _ in range(50):
    agent1.environment.seed = 0
    print(_)
    ep_reward, _ = agent1.train_agent(train=True)
    rewards1.append(ep_reward)
# agent1.epsilon = 0.0
# for _ in range(50):
#     agent1.environment.seed = 0
#     print(_)
#     ep_reward, _ = agent1.train_agent(train=True)
#     rewards1.append(ep_reward)
plt.plot(rewards1)
plt.show()

agent1.train_agent(train=False, render=True)

env2 = SimpleGridV2(size=size, agent_pos = torch.tensor([2, 2]), goal_pos = torch.tensor([17, 17]), device=device, render_mode='human')
env2.vertical_wall = False
env2.reset()
env2.render()

agent2 = DQNAgent(environment=env2,
                 alpha=1e-3,
                 epsilon=0.1,
                 gamma=0.9,
                 hidden_layer_sizes=(128, 128),
                 batch_size=100,
                 memory_length=int(1e6),
                 device=device)

rewards2 = []
for _ in range(50):
    agent1.environment.seed = 0
    print(_)
    ep_reward, _ = agent2.train_agent(train=True)
    rewards2.append(ep_reward)
# agent2.epsilon = 0.0
# for _ in range(50):
#     agent1.environment.seed = 0
#     print(_)
#     ep_reward, _ = agent2.train_agent(train=True)
#     rewards2.append(ep_reward)
plt.plot(rewards2)
plt.show()

agent2.train_agent(train=False, render=True)

eval_11 = []
eval_12 = []
eval_21 = []
eval_22 = []
for _ in range(50):
    print(_)
    ep_reward, _ = agent1.train_agent(train=False)
    eval_11.append(ep_reward)
    ep_reward, _ = agent2.train_agent(train=False)
    eval_22.append(ep_reward)

    agent1.environment = env2
    agent2.environment = env1

    ep_reward, _ = agent1.train_agent(train=False)
    eval_12.append(ep_reward)
    ep_reward, _ = agent2.train_agent(train=False)
    eval_21.append(ep_reward)

    agent1.environment = env1
    agent2.environment = env2

plt.scatter(range(50), eval_11, label='11')
plt.scatter(range(50), eval_12, label='11')
plt.scatter(range(50), eval_21, label='21')
plt.scatter(range(50), eval_22, label='22')
plt.legend()
plt.show()

agent1.environment = env2
agent2.environment = env1

agent1.train_agent(train=False, render=True)
agent2.train_agent(train=False, render=True)


plt.violinplot(positions=[0], dataset=eval_11)
plt.violinplot(positions=[1], dataset=eval_22)
plt.violinplot(positions=[2], dataset=eval_12)
plt.violinplot(positions=[3], dataset=eval_21)
plt.show()

min_r = min(eval_11 + eval_12 + eval_21 + eval_22)
max_r = max(eval_11 + eval_12 + eval_21 + eval_22)

sopr_11 = [(max_r - r) / (max_r - min_r) for r in eval_11]
sopr_12 = [(max_r - r) / (max_r - min_r) for r in eval_12]
sopr_21 = [(max_r - r) / (max_r - min_r) for r in eval_21]
sopr_22 = [(max_r - r) / (max_r - min_r) for r in eval_22]

plt.violinplot(positions=[0], dataset=sopr_11, showmedians=True, showextrema=False)
plt.violinplot(positions=[1], dataset=sopr_22, showmedians=True, showextrema=False)
plt.violinplot(positions=[2], dataset=sopr_21, showmedians=True, showextrema=False)
plt.violinplot(positions=[3], dataset=sopr_12, showmedians=True, showextrema=False)
plt.show()

dists_12 = []
for _ in range(50):
    next_states_a, rewards_a = env1.get_all_transitions(render=False)
    next_states_b, rewards_b = env2.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists_12.append(ot.emd2(a=a, b=b, M=M))
print("done")

dists_21 = []
for _ in range(50):
    next_states_a, rewards_a = env2.get_all_transitions(render=False)
    next_states_b, rewards_b = env1.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists_21.append(ot.emd2(a=a, b=b, M=M))
print("done")

np.mean(dists_12)
np.std(dists_12)
np.mean(dists_21)
np.std(dists_21)

plt.violinplot(positions=[0], dataset=sopr_12, showmedians=True, showextrema=False, bw_method=5e-1)
plt.violinplot(positions=[1], dataset=sopr_21, showmedians=True, showextrema=False, bw_method=5e-1)
plt.xticks(ticks = [0, 1], labels=['A-B', 'B-A'])
plt.xlabel("Direction of comparison")
plt.ylabel("SOPR (lower is better)")
plt.yticks(ticks=np.arange(0, 1.05, 0.1))
plt.title("Non-symmetry in SOPR, SimpleGrid")
plt.tight_layout()
plt.savefig('simplegrid_nonsymmetry_sopr.png', dpi=300)
plt.show()


np.mean(sopr_12)
np.std(sopr_12)

np.mean(sopr_21)
np.std(sopr_21)

env1.reset()
env1.render()

env2.reset()
env2.render()

agent2.environment = env2

agent2
