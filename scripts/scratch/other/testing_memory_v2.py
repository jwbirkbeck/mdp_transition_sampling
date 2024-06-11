import torch
from src.soft_actor_critic.memory_v2 import MemoryV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mem = MemoryV2(state_length=3, action_length=2, max_memories=10, device=device)

states = torch.cat((torch.arange(start=1, end=21).reshape(-1, 1),
                    torch.arange(start=1, end=21).reshape(-1, 1),
                    torch.arange(start=1, end=21).reshape(-1, 1)), axis=1)
actions = torch.cat((torch.arange(start=1, end=21).reshape(-1, 1),
                     torch.arange(start=1, end=21).reshape(-1, 1)), axis=1)
next_states = torch.cat((torch.arange(start=1, end=21).reshape(-1, 1),
                    torch.arange(start=1, end=21).reshape(-1, 1),
                    torch.arange(start=1, end=21).reshape(-1, 1)), axis=1)
rewards = torch.arange(1, 21).reshape(-1, 1)
terminateds = torch.arange(1, 21).reshape(-1, 1)

mem.append(states=states[0:7],
           actions=actions[0:7],
           next_states=next_states[0:7],
           rewards=rewards[0:7],
           terminateds=terminateds[0:7])

mem.append(states=states[8:12],
           actions=actions[8:12],
           next_states=next_states[8:12],
           rewards=rewards[8:12],
           terminateds=terminateds[8:12])
