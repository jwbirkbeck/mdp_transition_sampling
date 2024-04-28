import torch
import numpy as np
from src.policy_selector.sac_policy import SACPolicy
from src.soft_actor_critic.memory import SACMemory
from src.dyna_like.gaussian_world_model import GaussianWorldModel

class PolicySelector:
    def __init__(self, env, n_policies, device):
        self.env = env
        self.n_policies = n_policies
        self.device = device

        self.active_ind = 0

        self.hidden_layer_sizes = (128, 256)
        self.alpha = 1e-4
        self.gamma = 0.99
        self.batch_size = 1
        self.memory_length = 1
        self.polyak = 0.995

        self.agents = [SACPolicy(environment=env,
                                hidden_layer_sizes=self.hidden_layer_sizes,
                                alpha=self.alpha,
                                gamma=self.gamma,
                                batch_size=self.batch_size,
                                memory_length=self.memory_length,
                                device=self.device,
                                polyak=self.polyak) for _ in range(self.n_policies)]

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        self.wm_avgs_initialised = False
        self.wm_decaying_avgs = np.zeros(shape=(n_policies, ))
        self.wm_avg_decay = 0.95
        self.worldmodels = [GaussianWorldModel(state_size=state_size,
                                               action_size=action_size,
                                               alpha=5e-4,
                                               device=device,
                                               hidden_layer_sizes=(64, 64)) for _ in range(self.n_policies)]


    def execute_episode(self, method, train=True, render=False):
        assert method in ['episodic', 'decaying average'], 'Method must be either "episodic" or "decaying average"'
        self.env.reset()
        if method == 'episodic':
            # Execute for an episode and return the trajectory
            ep_rewards = ep_return = 0
            discount = 1.0
            done = truncated = False
            self.env.reset()
            mem = SACMemory(device=self.device, memory_length=501)
            while not (done or truncated):
                agent = self.agents[self.active_ind]
                state, action, next_state, reward, done, truncated, info = agent.single_step(train=train, render=render)
                mem.append(state=state.reshape(1, -1),
                           action=action.reshape(1, -1),
                           next_state=next_state.reshape(1, -1),
                           reward=torch.tensor([reward]).reshape(1, -1),
                           done=torch.tensor([done]).reshape(1, -1))
                ep_rewards += reward
                ep_return += reward * discount
                discount *= self.gamma
            w1_dists = []
            for wm in self.worldmodels:
                wm_preds = wm.sample(state=mem.state,
                                     action=mem.action,
                                     training=False)
                dist = wm.calc_w1_dist(predictions=wm_preds,
                                       next_states=mem.next_state,
                                       rewards=mem.reward)
                w1_dists.append(dist)
            best_worldmodel_index = w1_dists.index(min(w1_dists))
            old_active_ind = self.active_ind
            self.active_ind = best_worldmodel_index
            worldmodel = self.worldmodels[self.active_ind]
            if train:
                for s, a, ns, r, done in zip(mem.state, mem.action, mem.next_state, mem.reward, mem.done):
                    worldmodel.train_step(state=s.reshape(1, -1),
                                          action=a.reshape(1, -1),
                                          next_state=ns.reshape(1, -1),
                                          reward=r.reshape(1, -1))
            return ep_rewards, ep_return, old_active_ind, w1_dists

        # if method == 'episodic':
        #     # Execute for an episode and return the trajectory
        #     ep_rewards, ep_return = agent.train_agent(train=train, render=render)
        #     states = agent.memory.state
        #     actions = agent.memory.action
        #     next_states = agent.memory.next_state
        #     rewards = agent.memory.reward
        #     dones = agent.memory.done
        #     # Train the active worldmodel based on the episode
        #     if train:
        #         for s, a, ns, r, done in zip(states, actions, next_states, rewards, dones):
        #             worldmodel.train_step(state=s.reshape(1, -1),
        #                                   action=a.reshape(1, -1),
        #                                   next_state=ns.reshape(1, -1),
        #                                   reward=r.reshape(1, -1))
        #     # Select the new worldmodel based on wasserstein distance between episode and worldmodel:
        #     w1_dists = []
        #     for wm in self.worldmodels:
        #         wm_preds = wm.sample(state=states, action=actions, training=False)
        #         dist = wm.calc_w1_dist(predictions=wm_preds, next_states=next_states, rewards=rewards)
        #         w1_dists.append(dist)
        #
        #     best_worldmodel_index = w1_dists.index(min(w1_dists))
        #     self.active_ind = best_worldmodel_index
        #     return ep_rewards, ep_return, self.active_ind, w1_dists

        elif method == 'decaying average':
            ep_rewards = ep_return = 0
            discount = 1.0
            done = truncated = False
            self.env.reset()
            active_inds = None
            w1_dists = None
            while not (done or truncated):
                agent = self.agents[self.active_ind]
                state, action, next_state, reward, done, truncated, info = agent.single_step(train=train, render=render)

                for ind, wm in enumerate(self.worldmodels):
                    wm_preds = wm.sample(state=state.reshape(1, -1), action=action.reshape(1, -1), training=False)
                    dist = wm.calc_w1_dist(predictions=wm_preds,
                                           next_states=next_state.reshape(1, -1),
                                           rewards=torch.tensor([reward]).reshape(1, -1))
                    if not self.wm_avgs_initialised:
                        self.wm_decaying_avgs[ind] = dist
                    else:
                        self.wm_decaying_avgs[ind] *= self.wm_avg_decay
                        self.wm_decaying_avgs[ind] += (1 - self.wm_avg_decay) * dist
                self.wm_avgs_initialised = True
                best_worldmodel_index = np.argmin(self.wm_decaying_avgs)
                self.active_ind = best_worldmodel_index
                worldmodel = self.worldmodels[self.active_ind]
                worldmodel.train_step(state=state.reshape(1, -1),
                                      action=action.reshape(1, -1),
                                      next_state=next_state.reshape(1, -1),
                                      reward=torch.tensor([reward]).reshape(1, -1))
                if active_inds is None:
                    active_inds = np.array([[self.active_ind]])
                    w1_dists = np.array([self.wm_decaying_avgs])
                else:
                    active_inds = np.concatenate((active_inds, np.array([[self.active_ind]])))
                    w1_dists = np.concatenate((w1_dists, np.array([self.wm_decaying_avgs])))
                ep_rewards += reward
                ep_return += reward * discount
                discount *= self.gamma
            return ep_rewards, ep_return, active_inds, w1_dists