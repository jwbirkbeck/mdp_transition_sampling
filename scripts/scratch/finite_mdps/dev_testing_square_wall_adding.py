import torch
from src.grid_worlds.simple_grid_v2 import SimpleGridV2

size = 20
device = torch.device('cpu')
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')
env.render()

env._add_walls_around_agent()
