import torch
from src.finite_mdps.torch_helpers import TorchBox
import gymnasium as gym
import pygame


class SimpleGridV2(gym.Env):
    def __init__(self, size, device, seed=None, render_mode=None, reward_func='manhattan', agent_pos=None, goal_pos=None):
        super().__init__()
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        assert isinstance(size, int) and isinstance(size, int)
        assert size >= 10
        assert isinstance(device, torch.device)
        assert render_mode in ["human", "rgb_array"]
        assert reward_func in ["manhattan", "euclidean", "sparse"]

        if agent_pos is not None and goal_pos is not None:
            assert seed == None
            assert isinstance(agent_pos, torch.Tensor)
            assert isinstance(goal_pos, torch.Tensor)
            self._init_agent_pos = agent_pos
            self._init_goal_pos = goal_pos
            self.seeded = False
        else:
            assert isinstance(seed, int)
            self._init_agent_pos = None
            self._init_goal_pos = None
            self.seeded = True

        self.size = size
        self.device = device
        self.seed = seed
        self.render_mode = render_mode
        self.reward_func = reward_func

        self._agent_pos = None
        self._goal_pos = None

        self._timestep = 0
        self._max_timestep = 100

        self.windy = False
        self.wind_prob = torch.tensor([0.0], device=self.device, requires_grad=False)

        self.walls_around_agent = False
        self.vertical_wall = False

        low = torch.zeros(size=(4, ), dtype=torch.int, device=self.device, requires_grad=False)
        high = size * torch.ones(size=(4, ), dtype=torch.int, device=self.device, requires_grad=False)
        self.observation_space = TorchBox(low=low, high=high)
        self.action_space = gym.spaces.Discrete(4) # north, east, south, west

        self._action_map = {0: torch.tensor([ 0, -1], dtype=torch.int, device=self.device, requires_grad=False),
                            1: torch.tensor([ 1,  0], dtype=torch.int, device=self.device, requires_grad=False),
                            2: torch.tensor([ 0,  1], dtype=torch.int, device=self.device, requires_grad=False),
                            3: torch.tensor([-1,  0], dtype=torch.int, device=self.device, requires_grad=False)}

        self._object_tensor_map = {0: torch.tensor([0], dtype=torch.int, device=self.device, requires_grad=False),
                                   1: torch.tensor([1], dtype=torch.int, device=self.device, requires_grad=False),
                                   2: torch.tensor([2], dtype=torch.int, device=self.device, requires_grad=False),
                                   3: torch.tensor([3], dtype=torch.int, device=self.device, requires_grad=False)}

        self._object_map = {'space': 0, 'agent': 1, 'goal': 2, 'wall': 3}

        self._reset_rng(seed=self.seed)
        self._initialise()

        self.reward_scaling = True
        self.min_reward = self.max_reward = None
        if self.reward_func == 'euclidean':
            self._get_reward = self._get_reward_euclidean
        elif self.reward_func == 'manhattan':
            self._get_reward = self._get_reward_manhattan
        elif self.reward_func == 'sparse':
            self._get_reward = self._get_reward_sparse
        else:
            raise NotImplementedError
        self._set_reward_scalers()

        self._pygame_window = None
        self._pygame_window_width = None
        self._pygame_window_height = None
        self._pygame_clock = None
        self._pygame_square_pix = 20

    def step(self, action):
        assert self._timestep <= self._max_timestep, 'episode has terminated'
        
        terminated = False
        truncated = False
        info = {}
        
        next_grid, reward = self._update_agent_pos(action=action)
        self.grid = next_grid
        self._timestep += 1

        observation = self._get_obs()

        # if torch.all(self.agent_pos == next_state).item():
        #     reward -= 1.0

        if self._timestep == self._max_timestep:
            terminated = True
        
        return observation, reward, terminated, truncated, info

    def get_all_transitions(self, render=False):
        width_range = range(self.size)
        height_range = range(self.size)
        next_states_flat = torch.tensor([], device=self.device)
        rewards_flat = torch.tensor([], device=self.device)
        transition_ind = 0
        for row in width_range:
            for col in height_range:
                if self.grid[row, col] != self._object_map['wall']:
                    for action in self._action_map:
                        self._set_state_for_transitions(row=row, col=col)
                        if render:
                            self.render()
                        observation, reward, terminated, truncated, info = self.step(action=action)
                        if render:
                            self.render()
                        self._timestep -= 1  # Undo the timestep iteration in self.step when getting transitions
                        # next_states_flat[transition_ind, :] = observation
                        # rewards_flat[transition_ind, :] = reward
                        next_states_flat = torch.cat((next_states_flat, observation.reshape(1, -1)), dim=0)
                        rewards_flat = torch.cat((rewards_flat, reward.reshape(1, -1)), dim=0)
                        transition_ind += 1
        return next_states_flat, rewards_flat

    def reset(self, **kwargs):
        self._reset_rng(seed=self.seed)
        self._initialise()
        self._timestep = 0
        if self.walls_around_agent:
            self._add_walls_around_agent()
        if self.vertical_wall:
            self._add_vertical_wall()
        return self._get_obs(), {}

    def _reset_rng(self, seed=None):
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed=self.seed)

    def _initialise(self):
        grid = torch.zeros(size=(self.size, self.size), dtype=torch.float, device=self.device, requires_grad=False)
        self._agent_pos = self._init_agent_pos = self._reset_agent_pos()
        grid[self._agent_pos[0], self._agent_pos[1]] = self._object_map['agent']
        self._goal_pos = self._init_goal_pos = self._reset_goal_pos()
        grid[self._goal_pos[0], self._goal_pos[1]] = self._object_map['goal']
        grid[0, :] = self._object_map['wall']
        grid[-1, :] = self._object_map['wall']
        grid[:, 0] = self._object_map['wall']
        grid[:, -1] = self._object_map['wall']
        self.grid = grid

    def _reset_agent_pos(self):
        if not self.seeded:
            agent_pos = self._init_agent_pos
        else:
            pos_x = torch.randint(low=1, high=self.size-1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            pos_y = torch.randint(low=1, high=int(self.size/4), size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            agent_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return agent_pos

    def _reset_goal_pos(self):
        if not self.seeded:
            goal_pos = self._init_goal_pos
        else:
            pos_x = torch.randint(low=1, high=self.size - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            pos_y = torch.randint(low=int(3 * self.size / 4), high=self.size - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            goal_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return goal_pos

    def _update_agent_pos(self, action):
        curr_agent_pos = self._agent_pos
        if self.windy and (torch.rand((1, )) < self.wind_prob):
            action_tensor = self._action_map[torch.randint(low=0, high=len(self._action_map), size=(1, )).item()]
        else:
            action_tensor = self._action_map[action]

        new_agent_pos = curr_agent_pos + action_tensor
        next_grid = self.grid.clone().detach()

        if self.grid[new_agent_pos[0], new_agent_pos[1]] == self._object_tensor_map[3]:
            new_agent_pos = curr_agent_pos
        else:
            next_grid[curr_agent_pos[0], curr_agent_pos[1]] = self._object_map['space']
            next_grid[new_agent_pos[0], new_agent_pos[1]] = self._object_map['agent']

        self._agent_pos = new_agent_pos

        reward = self._get_reward(agent_pos=new_agent_pos, goal_pos=self._goal_pos)
        if self.reward_scaling:
            reward = (reward - self.min_reward) / (self.max_reward - self.min_reward) - 1.0
        return next_grid, reward

    def _get_obs(self):
        obs = torch.cat((self._agent_pos, self._goal_pos), dim = 0)
        return obs

    def _get_reward_manhattan(self, **kwargs):
        assert kwargs['agent_pos'] is not None
        assert kwargs['goal_pos'] is not None
        agent_pos = kwargs['agent_pos']
        goal_pos = kwargs['goal_pos']
        reward = -1.0 * torch.sum(torch.abs((agent_pos - goal_pos)).to(torch.float))
        return reward

    def _get_reward_euclidean(self, **kwargs):
        assert kwargs['agent_pos'] is not None
        assert kwargs['goal_pos'] is not None
        agent_pos = kwargs['agent_pos']
        goal_pos = kwargs['goal_pos']
        reward = -1.0 * torch.linalg.norm((agent_pos - goal_pos).to(torch.float))
        return reward

    def _get_reward_sparse(self, **kwargs):
        assert kwargs['agent_pos'] is not None
        assert kwargs['goal_pos'] is not None
        agent_pos = kwargs['agent_pos']
        goal_pos = kwargs['goal_pos']
        if torch.equal(agent_pos, goal_pos):
            reward = torch.tensor([1.0], device=self.device, requires_grad=False)
        else:
            reward = torch.tensor([0.0], device=self.device, requires_grad=False)
        return reward

    def _set_reward_scalers(self):
        min = 999
        max = -999
        for pos in [torch.tensor([1., 1.]),
                    torch.tensor([self.size - 1, 1.]),
                    torch.tensor([1., self.size - 1.]),
                    torch.tensor([self.size - 1., self.size - 1.]),
                    self._goal_pos]:
            curr_reward = self._get_reward(agent_pos=pos, goal_pos=self._goal_pos)
            max = curr_reward if curr_reward > max else max
            min = curr_reward if curr_reward < min else min
        self.min_reward = min
        self.max_reward = max

    def _get_walls(self):
        grid = self.grid
        walls = (grid == 3)
        wall_indices = walls.nonzero().tolist()
        return wall_indices

    def _add_walls_around_agent(self):
        # constructs a 3x3 grid around the agent position
        left_wall_x = self._agent_pos[0] - 2
        right_wall_x = self._agent_pos[0] + 2
        top_wall_y = self._agent_pos[1] - 2
        bottom_wall_y = self._agent_pos[1] + 2

        # left wall:
        if left_wall_x > 0 and left_wall_x < self.size:
            for y in range(top_wall_y, bottom_wall_y+1):
                if y > 0 and y < self.size:
                    self.grid[left_wall_x, y] = self._object_map['wall']
        # right wall
        if right_wall_x > 0 and right_wall_x < self.size:
            for y in range(top_wall_y, bottom_wall_y+1):
                if y > 0 and y < self.size:
                    self.grid[right_wall_x, y] = self._object_map['wall']
        # top wall
        if top_wall_y > 0 and top_wall_y < self.size:
            for x in range(left_wall_x, right_wall_x + 1):
                if x > 0 and x < self.size:
                    self.grid[x, top_wall_y] = self._object_map['wall']
        # bottom wall
        if bottom_wall_y > 0 and bottom_wall_y < self.size:
            for x in range(left_wall_x, right_wall_x + 1):
                if x > 0 and x < self.size:
                    self.grid[x, bottom_wall_y] = self._object_map['wall']

    def _add_vertical_wall(self):
        x = 10
        for y in range(2, self.size):
            self.grid[x, y] = self._object_map['wall']

    def _set_state_for_transitions(self, row, col):
        self.grid[self._goal_pos[0], self._goal_pos[1]] = self._object_map['goal']
        self.grid[self._agent_pos[0], self._agent_pos[1]] = self._object_map['space']
        self.grid[row, col] = self._object_map['agent']
        self._agent_pos = torch.tensor([row, col], dtype=torch.int, device=self.device, requires_grad=False)

    def render(self):
        if self._pygame_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._pygame_window_width = self._pygame_square_pix * self.size
            self._pygame_window_height = self._pygame_square_pix * self.size
            window = pygame.display.set_mode((self._pygame_window_width, self._pygame_window_height))
            self._pygame_window = window
        if self._pygame_clock is None and self.render_mode == "human":
            self._pygame_clock = pygame.time.Clock()
        canvas = pygame.Surface((self._pygame_window_width, self._pygame_window_height))
        canvas.fill((255, 255, 255))

        # # #
        # agent
        # # #
        agent_center = (self._pygame_square_pix * (self._agent_pos[0].item() + 0.5),
                        self._pygame_square_pix * (self._agent_pos[1].item() + 0.5))

        pygame.draw.circle(surface=canvas, color=(0, 0, 255), center=agent_center, radius=self._pygame_square_pix/1.9)

        # # #
        # goal
        # # #
        rect = pygame.Rect(self._pygame_square_pix * (self._goal_pos[0].item()),
                           self._pygame_square_pix * (self._goal_pos[1].item()),
                           self._pygame_square_pix, self._pygame_square_pix)

        pygame.draw.rect(surface=canvas, color=(255, 0, 0), rect=rect)

        # # #
        # walls
        # # #
        wall_indices = self._get_walls()
        for idx in wall_indices:
            rect = pygame.Rect(self._pygame_square_pix * (idx[0]), self._pygame_square_pix * (idx[1]),
                               self._pygame_square_pix, self._pygame_square_pix)
            pygame.draw.rect(surface=canvas, color=(128, 128, 128), rect=rect)

        # # #
        # gridlines
        # # #
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, self._pygame_square_pix * x),
                             (self._pygame_square_pix * self.size, self._pygame_square_pix * x), width=1, )
        for y in range(self.size + 1):
            pygame.draw.line(canvas, 0, (self._pygame_square_pix * y, 0),
                             (self._pygame_square_pix * y, self._pygame_square_pix * self.size), width=1, )

        # # #
        # render
        # # #
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._pygame_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._pygame_clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_window is not None:
            pygame.display.quit()
            pygame.quit()
            self._pygame_window = None
            self._pygame_clock = None

    def get_manual_step(self, state, action):
        self._timestep -= 1  # Avoid counting this as a valid step against the episode time limit
        new_agent_pos = state[0:2, ]
        # new_goal_pos = state[2:4, ]
        self._agent_pos = torch.tensor(new_agent_pos, dtype=torch.int, device=self.device, requires_grad=False)
        # self._goal_pos = torch.tensor(new_goal_pos, dtype=torch.int, device=self.device, requires_grad=False)
        observation, reward, terminated, truncated, info = self.step(action=action.item())
        return observation, reward, terminated, truncated, info

    def compute_reward_wrap(self, states, actions):
        rewards = []
        for state, action in zip(states, actions):
            _, reward, _, _, _ = self.get_manual_step(state=state, action=action)
            rewards.append(reward)
        return rewards

    def sample_reward_function(self, n_samples):
        return -1.0 * torch.rand(size=(n_samples, ), requires_grad=False)
