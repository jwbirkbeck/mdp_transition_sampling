import numpy as np
import ot


class RandomSampler:
    def __init__(self, state_space, action_space, env_reward_function):
        self.state_space = state_space
        self.action_space = action_space
        self.env_reward_function = env_reward_function

    def sample(self, s_reward):
        s_sample = self.state_space.sample()
        a_sample = self.action_space.sample()
        converged = True
        # TODO: random sample from action space
        return s_sample, a_sample, converged


class MCCESampler:
    def __init__(self, state_space, action_space, env_reward_function):
        self.state_space = state_space
        self.action_space = action_space
        self.env_reward_function = env_reward_function
        self.max_iterations = 100
        self.mcce_sample_size = 40
        self.mcce_sample_size_elite = 4
        self.mcce_min_diff = 1e-3
        self.initial_sigma_coefficient = 0.25

    def get_init_params(self, bounds):
        half_range = (bounds.high - bounds.low) / 2.0
        midpoint = bounds.high - half_range
        mu = np.array([midpoint])
        sigma = half_range * self.initial_sigma_coefficient
        return mu, sigma

    def sample(self, s_reward):
        state_length = self.state_space.low.shape[0]
        state_mu, state_sigma = self.get_init_params(bounds=self.state_space)
        action_mu, action_sigma = self.get_init_params(bounds=self.action_space)
        mu = np.append(state_mu, action_mu, axis=1)
        sigma = np.append(state_sigma, action_sigma, axis=0)
        converged = False
        iterations = 0
        while not converged and iterations <= self.max_iterations:
            iterations += 1
            samples = np.random.normal(loc=mu, scale=sigma, size=(self.mcce_sample_size, mu.shape[1]))
            samples = np.clip(samples,
                              a_min=np.append(self.state_space.low, self.action_space.low),
                              a_max=np.append(self.state_space.high, self.action_space.high))
            states = samples[:, :state_length]
            actions = samples[:, state_length:]
            reward_error = np.abs(np.subtract(s_reward, self.env_reward_function(states=states, actions=actions)))
            elite_sample_inds = np.argpartition(-reward_error, -self.mcce_sample_size_elite)[
                                -self.mcce_sample_size_elite:]
            elite_sample = samples[elite_sample_inds, :]
            mu = np.mean(elite_sample, axis=0, keepdims=True)
            sigma = np.std(elite_sample, axis=0, keepdims=True) + 1e-6
            if all(reward_error < self.mcce_min_diff):
                converged = True
        final_sample = np.random.normal(loc=mu, scale=sigma, size=(1, mu.shape[1]))
        final_state = final_sample[:, :state_length]
        final_action = final_sample[:, state_length:]
        return final_state, final_action, converged

class MDPDifferenceSampler:
    def __init__(self, environment_a, environment_b, state_space, action_space, method='mcce'):
        self.environment_a = environment_a
        self.environment_b = environment_b
        self.state_space = state_space
        self.action_space = action_space

        self.method = method
        if self.method == 'mcce':
            self.sampler = MCCESampler(state_space=self.state_space, action_space=self.action_space,
                                       env_reward_function=self.environment_a.compute_reward_wrap)
        elif self.method =='random':
            self.sampler = RandomSampler(state_space=self.state_space, action_space=self.action_space,
                                         env_reward_function=self.environment_a.compute_reward_wrap)

    def get_difference(self, n_states, n_transitions):
        state_ncols = self.environment_a.observation_space.low.shape[0]
        observations_a = np.empty(shape=(n_states * n_transitions, state_ncols + 1))
        observations_b = np.empty(shape=(n_states * n_transitions, state_ncols + 1))
        s_states, s_actions, _, _ = self.sample_states_and_actions(n_samples=n_states)
        for ind in range(s_states.shape[0]):
            state = s_states[ind]
            action = s_actions[ind]
            obs_a = self.sample_transitions(mdp='a', state=state, action=action, n_transitions=n_transitions)
            obs_b = self.sample_transitions(mdp='b', state=state, action=action, n_transitions=n_transitions)
            observations_a[ind*n_transitions:(ind+1)*n_transitions, :] = obs_a
            observations_b[ind*n_transitions:(ind+1)*n_transitions, :] = obs_b
        wasserstein_1 = self.get_wasserstein_1(samples_a=observations_a[[0]], samples_b=observations_b[[0]])
        return wasserstein_1

    def sample_rewards(self, n_samples):
        rewards = self.environment_a.sample_reward_function(n_samples=n_samples)
        return rewards

    def sample_states_and_actions(self, n_samples):
        s_rewards = self.sample_rewards(n_samples=n_samples)
        s_states = np.empty(shape=(n_samples, self.state_space.low.shape[0]))
        s_actions = np.empty(shape=(n_samples, self.action_space.shape[0]))
        states_converged = [None] * n_samples
        actions_converged = [None] * n_samples
        for ind in range(len(s_rewards)):
            state, action, converged = self.sampler.sample(s_reward=s_rewards[ind])
            s_states[ind] = state
            s_actions[ind] = action
            states_converged[ind] = converged
            actions_converged[ind] = converged
        return s_states, s_actions, states_converged, actions_converged

    @staticmethod
    def get_wasserstein_1(samples_a, samples_b):
        a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
        b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
        M = ot.dist(samples_a, samples_b, metric='euclidean', p=1)
        return ot.emd2(a=a, b=b, M=M)

    def sample_transitions(self, mdp, state, action, n_transitions):
        assert mdp in ('a', 'b')
        env = self.environment_a if mdp == 'a' else self.environment_b
        next_states = np.empty(shape=(n_transitions, state.shape[0]))
        rewards = np.empty(shape=(n_transitions, 1))
        for ind in range(n_transitions):
            env.reset()
            next_state, reward, done, truncated, info = env.get_manual_step(state=state, action=action)
            next_states[ind] = next_state
            rewards[ind] = reward
        observation = np.hstack((next_states, rewards))
        return observation

