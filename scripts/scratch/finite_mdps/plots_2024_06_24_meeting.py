import torch
import matplotlib.pyplot as plt
from src.grid_worlds.simple_grid_v1 import SimpleGridV1

# Need plots for:
    # Base MDP
    # Positional variants
    # Additional random wall variants

device = torch.device('cpu')
env = SimpleGridV1(height=16, width=16, seed=0, device=device, render_mode='human')

env.reset()
env.render()

env.seed = 1

env._add_random_walls(2)


# Set an env's initial goal and agent pos manually, measure MDP difference, deploy trained agent, plot diffs
