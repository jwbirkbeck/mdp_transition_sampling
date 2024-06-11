import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dyna_like_agent.gaussian_world_model import GaussianWorldModel
from src.metaworld.wrapper import MetaWorldWrapper

env = MetaWorldWrapper(task_name='reach-v2')

state_size=env.observation_space.shape[0]
action_size = env.action_space.shape[0]
alpha = 1e-3
device = torch.device('cpu')
hidden_layer_sizes = (64, 64)


worldmodel = GaussianWorldModel(state_size=state_size,
                                action_size=action_size,
                                alpha=alpha,
                                device=device,
                                hidden_layer_sizes=hidden_layer_sizes)

decay = 0.99
lo = 9e9
hi = -9e9
losses = []
lows = []
highs = []
done = truncated = False
env.reset()
for _ in range(2500):
    state = env.get_observation()
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    state = torch.Tensor(state).reshape(1, -1)
    action = torch.Tensor(action).reshape(1, -1)
    next_state = torch.Tensor(next_state).reshape(1, -1)
    reward_true = torch.Tensor([reward]).reshape(1, -1)

    this_loss = worldmodel.train_step(state=state, action=action, next_state=next_state, reward=reward_true)
    losses.append(this_loss)
    lo *= (2 - decay)
    hi *= decay
    lo = this_loss if this_loss < lo else lo
    hi = this_loss if this_loss > hi else hi
    lows.append(lo)
    highs.append(hi)
    if done or truncated:
        env.change_task()
print("done")

plt.plot(losses, label = "losses")
plt.plot(lows, label ="lows")
plt.plot(highs, label ="highs")
# plt.plot(np.log(bound_range), label = "range")
plt.legend()
plt.xlim(1500, 2000)
plt.ylim(0, 0.1)
plt.show()

# For a base batch size of N:
# Execute N steps to fill out the true batch, recording the transitions
# Calculate the accuracy of the trajectory of length N
# Use accuracy ratio to determine true:simulated. e.g. for a ratio of 0.5, use 32 real and 32 simulated transitions
# use the Wasserstein distance of the recent trajectory as a ratio between simulated and real transitions in the batch

losses = np.array(losses)
lows = np.array(lows)
highs = np.array(highs)

numerator = (highs - losses)
denominator = (highs - lows)
denominator[denominator==0] = np.nan
final = numerator / denominator
final[np.isnan(final)] = 0

base_batch_size = np.ones(shape=(len(final), )) * 32
simulated_transitions = np.round(base_batch_size * final)
total_transitions = base_batch_size + simulated_transitions

sum(simulated_transitions) / sum(base_batch_size)

plt.plot(simulated_transitions[0:500])
plt.show()

plt.plot(losses)
plt.show()



# from numpy.lib.stride_tricks import sliding_window_view
# plt.plot(np.max(sliding_window_view(losses, window_shape = 500), axis = 1))
# plt.plot(losses)
# plt.plot(np.min(sliding_window_view(losses, window_shape = 500), axis = 1))
# plt.show()

# When the difference between the losses and the high is large,, maximally use the world model
# When the difference between the lows and the loss is zero, maximal usage of world model
numerator = losses - lo
denominator = highs - lows
numerator[numerator == 0] += 1e-6
denominator[denominator == 0] = np.nan


plt.plot(numerator/denominator)
plt.show()

observation = worldmodel.sample(state=state, action=action, training=False)
observation - torch.cat((next_state, reward), dim=1)
