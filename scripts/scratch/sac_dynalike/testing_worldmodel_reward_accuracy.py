import torch
import matplotlib.pyplot as plt
from src.dyna_like_agent.gaussian_world_model import GaussianWorldModel
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool

# Train a worldmodel step by step
# Test it by randomly sampling states and actions

env = MetaWorldWrapper(task_name=task_pool[0])
env_manual = MetaWorldWrapper(task_name=task_pool[0])

state_size = env.observation_space.shape[0]
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
low = 9e9
high = -9e9
wm_losses = []
manual_losses = []
lows = []
highs = []
done = truncated = False
env.change_task(task_name=env.task_name, task_number=0)
env_manual.change_task(task_name=env.task_name, task_number=0)
for _ in range(2500):
    state = env.get_observation()
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    state = torch.Tensor(state).reshape(1, -1)
    action = torch.Tensor(action).reshape(1, -1)
    next_state = torch.Tensor(next_state).reshape(1, -1)
    reward_true = torch.Tensor([reward]).reshape(1, -1)

    this_wm_loss = worldmodel.train_step(state=state, action=action, next_state=next_state, reward=reward_true)
    wm_losses.append(this_wm_loss)
    low *= (2 - decay)
    high *= decay
    low = this_wm_loss if this_wm_loss < low else low
    high = this_wm_loss if this_wm_loss > high else high
    lows.append(low)
    highs.append(high)

    env_manual.reset()
    manual_next_state, manual_reward, _, _, _ = env_manual.get_manual_step(state=state.numpy().flatten(), action=action.numpy().flatten())
    manual_next_state = torch.Tensor(manual_next_state).reshape(1, -1)
    manual_reward = torch.Tensor([manual_reward]).reshape(1, -1)
    true_obs = torch.cat((next_state, reward_true), dim=1)
    manual_obs = torch.cat((manual_next_state, manual_reward), dim=1)
    manual_loss = torch.nn.L1Loss()(true_obs, manual_obs)
    manual_losses.append(manual_loss)
    if done or truncated:
        env.change_task(task_name=env.task_name, task_number=0)
        env_manual.change_task(task_name=env.task_name, task_number=0)
print("done")

plt.plot(wm_losses, label = "worldmodel losses")
# plt.plot(lows, label ="lows")
# plt.plot(highs, label ="highs")
plt.legend()
plt.show()

plt.plot(manual_losses, label = "manual step losses")
plt.legend()
plt.show()

plt.hist(manual_obs[:, :-1] - next_state)
plt.show()

env.reset()
state = torch.Tensor(env.observation_space.sample()).reshape(1, -1)
action =  torch.Tensor(env.action_space.sample()).reshape(1, -1)
wm_pred = worldmodel.sample(state=state, action=action, training=False)
pred_next_state, pred_reward = wm_pred[:, :-1], wm_pred[:, -1]

next_state, reward, _, _, _ = env.get_manual_step(state=state.numpy()[0,], action=action.numpy()[0,])
print(pred_reward - reward)
plt.hist(pred_next_state - next_state)
plt.show()

print(worldmodel.evaluate(state, action, torch.Tensor(next_state).reshape(1, -1), torch.Tensor([reward]).reshape(1, -1)))

wm_losses[-1]
