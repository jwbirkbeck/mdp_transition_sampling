from src.finite_mdps.simple_grid_v2 import SimpleGridV2
import torch
import numpy as np
import ot

size = 20
device = torch.device('cpu')

def get_true_distance(env_a, env_b):
    next_states_a, rewards_a = env_a.get_all_transitions(render=False)
    next_states_b, rewards_b = env_b.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    return ot.emd2(a=a, b=b, M=M)


class IncrementVector:
    def __init__(self):
        self.agent = torch.tensor([1, 1], requires_grad=False, dtype=torch.int32)
        self.goal = torch.tensor([18, 18], requires_grad=False, dtype=torch.int32)
        self.agents_turn_for_update = True

    def increment(self):
        if self.agents_turn_for_update:
            self.agents_turn_for_update = False
            if self.agent[0] < 18:
                self.agent[0] += 1
            elif self.agent[1] < 18:
                self.agent[0] = 1
                self.agent[1] += 1
            else:
                raise NotImplementedError
        else:
            self.agents_turn_for_update = True
            if self.goal[0] > 1:
                self.goal[0] -= 1
            elif self.goal[1] > 1:
                self.goal[0] = 18
                self.goal[1] -= 1
            else:
                raise NotImplementedError

pos_vector = IncrementVector()
for _ in range(5000):
    print(_)
    # print(f"{list(pos_vector.agent.numpy())}, {list(pos_vector.goal.numpy())}")
    pos_vector.increment()


pos_vector.agent
pos_vector.goal

pos_vector.increment()

dists = []
env_0 = SimpleGridV2(size=size, device=device, agent_pos=torch.tensor([1, 1]), goal_pos=torch.tensor([18, 18]), render_mode='human')
pos_vector = IncrementVector()
n_limit = 50
reps = 0
while reps < n_limit:
    print(reps)
    reps += 1
    pos_vector.increment()
    env_1 = SimpleGridV2(size=size, device=device, agent_pos=pos_vector.agent, goal_pos=pos_vector.goal, render_mode='human')
    dists.append(get_true_distance(env_0, env_1))
print("done")

import matplotlib.pyplot as plt
plt.plot(dists)
plt.show()

plt.plot(np.sort(np.array(dists)))
plt.show()

env_a = SimpleGridV2(size=size, device=device, agent_pos=torch.tensor([1, 1]), goal_pos=torch.tensor([18, 18]), render_mode='human')
env_b = SimpleGridV2(size=size, device=device, agent_pos=torch.tensor([18, 1]), goal_pos=torch.tensor([18, 18]), render_mode='human')

next_states_a, rewards_a = env_a.get_all_transitions(render=False)
next_states_b, rewards_b = env_b.get_all_transitions(render=False)
samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
ot.emd2(a=a, b=b, M=M)


env_b._init_agent_pos

