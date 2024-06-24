import torch
import matplotlib.pyplot as plt
from src.finite_mdps.simple_grid_v1 import SimpleGridV1
from src.dqn.dqn_agent import DQNAgent

device = torch.device('cpu')
env = SimpleGridV1(height=16, width=16, seed=0, device=device, render_mode='human')

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
    for _ in range(400):
        # agent.environment.seed = _ % 5
        # print(_)
        # ep_reward, _ = agent.train_agent(train=True)
        # rewards.append(ep_reward)
        agent.environment.seed = 5
        print(_)
        ep_reward, _ = agent.train_agent(train=False)
        rewards.append(ep_reward)
    print('done')

tester()

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylim(-2000, 0)
plt.ylabel('Episode total reward')
plt.vlines(x=400, ymin=-2000, ymax=0, linestyles='dashed', color='grey', alpha=0.5)
plt.title('DQN performance, SimpleGridV1')
plt.tight_layout()
plt.savefig('example_change_performance_drop.png')
plt.show()

agent.environment.seed=1
agent.environment.reset()
agent.environment.render()

env.seed = torch.randint(low=0, high=250, size=(1,)).item()
agent.train_agent(train=False, render=True)

"""
TODO: 
    Measure full MDP distance between two MDPs
    Use sampling approach to estimate this distance
    Compare estimates to true distances
    
    To calculate full distance:
        Initialise MDPs A, B
        Set the agent position to every possible state in both MDPs
        Execute all possible actions, storing resulting transitions
        Calculate Wasserstein distance using every transition
    
    Shortcut:
        Determine all possible unique locations in space (hard-coded)
        Determine all resulting transitions for each unique location
        Map reward function to each unique state
        Calculate Wasserstein distance across transitions
    
    For a simple grid:
        Centres
            No blocked movements
        Walls
            One blocked movement
        Corners
            Two blocked movements
        Reward function: 
            Cheaply calculable from goal position? 
"""

