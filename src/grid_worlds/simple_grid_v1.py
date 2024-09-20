import torch
from src.grid_worlds.torch_helpers import TorchBox
import gymnasium as gym
import pygame


class SimpleGridV1(gym.Env):

    def __init__(self, width, height, device, seed=None, render_mode=None, reward_func='manhattan', agent_pos=None, goal_pos=None):
        super().__init__()
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        assert isinstance(width, int) and isinstance(height, int)
        assert width >= 10 and height >= 10
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

        self.width = width
        self.height = height
        self.device = device
        self.seed = seed
        self.render_mode = render_mode
        self.reward_func = reward_func

        self._agent_pos = None
        self._goal_pos = None

        self._timestep = 0
        self._max_timestep = 100

        self.n_random_walls = 0
        self.windy = False
        self.wind_prob = torch.tensor([0.1], device=self.device, requires_grad=False)

        low = torch.zeros(size=(self.width * self.height, ), dtype=torch.int, device=self.device, requires_grad=False)
        high = 3 * torch.ones(size=(self.width * self.height, ), dtype=torch.int, device=self.device, requires_grad=False)
        self.observation_space = TorchBox(low=low, high=high)
        self.action_space = gym.spaces.Discrete(4) # north, east, south, west, nop

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

        next_state, reward = self._update_agent_pos(state=self.grid, action=action)
        self._timestep += 1

        if torch.all(self.grid == next_state).item():
            reward -= 1.0

        # if self._timestep == self._max_timestep:
        #     truncated = True

        if self._timestep == self._max_timestep:
            terminated = True

        self.grid = next_state
        observation = self.grid.flatten()

        # if torch.all(self._agent_pos == self._goal_pos).item():
        #     terminated = True

        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._pygame_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._pygame_window_width = self._pygame_square_pix * self.width
            self._pygame_window_height = self._pygame_square_pix * self.height
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
        for x in range(self.width + 1):
            pygame.draw.line(canvas, 0, (0, self._pygame_square_pix * x),
                             (self._pygame_square_pix * self.width, self._pygame_square_pix * x), width=1, )
        for y in range(self.height + 1):
            pygame.draw.line(canvas, 0, (self._pygame_square_pix * y, 0),
                             (self._pygame_square_pix * y, self._pygame_square_pix * self.height), width=1, )

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

    def reset(self, **kwargs):
        self._reset_rng(seed=self.seed)
        self._initialise()
        self._timestep = 0
        return self.grid, {}

    def _reset_rng(self, seed=None):
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed=self.seed)

    def _initialise(self):
        grid = torch.zeros(size=(self.width, self.height), dtype=torch.float, device=self.device, requires_grad=False)
        self._agent_pos = self._init_agent_pos = self._reset_agent_pos()
        grid[self._agent_pos[0], self._agent_pos[1]] = self._object_map['agent']
        self._goal_pos = self._init_goal_pos = self._reset_goal_pos()
        grid[self._goal_pos[0], self._goal_pos[1]] = self._object_map['goal']
        grid[0, :] = self._object_map['wall']
        grid[-1, :] = self._object_map['wall']
        grid[:, 0] = self._object_map['wall']
        grid[:, -1] = self._object_map['wall']
        self.grid = grid
        self._add_random_walls(n_walls=self.n_random_walls)

    def _reset_agent_pos(self):
        if not self.seeded:
            agent_pos = self._init_agent_pos
        else:
            pos_x = torch.randint(low=1, high=self.width-1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            pos_y = torch.randint(low=1, high=int(self.height/4), size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            agent_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return agent_pos

    def _reset_goal_pos(self):
        if not self.seeded:
            goal_pos = self._init_goal_pos
        else:
            pos_x = torch.randint(low=1, high=self.width - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            pos_y = torch.randint(low=int(3 * self.height / 4), high=self.height - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
            goal_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return goal_pos

    def _update_agent_pos(self, state, action):
        curr_agent_pos = self._agent_pos
        if self.windy and (torch.rand((1, )) < self.wind_prob):
            action_tensor = self._action_map[torch.randint(low=0, high=len(self._action_map), size=(1, )).item()]
        else:
            action_tensor = self._action_map[action]

        new_agent_pos = curr_agent_pos + action_tensor
        next_state = state.clone().detach()

        if state[new_agent_pos[0], new_agent_pos[1]] == self._object_tensor_map[3]:
            new_agent_pos = curr_agent_pos
        else:
            next_state[curr_agent_pos[0], curr_agent_pos[1]] = self._object_map['space']
            next_state[new_agent_pos[0], new_agent_pos[1]] = self._object_map['agent']

        self._agent_pos = new_agent_pos

        reward = self._get_reward(agent_pos=new_agent_pos, goal_pos = self._goal_pos)
        if self.reward_scaling:
            reward = (reward - self.min_reward) / (self.max_reward - self.min_reward) - 1.0
        return next_state, reward

    def _get_walls(self):
        grid = self.grid
        walls = (grid == 3)
        wall_indices = walls.nonzero().tolist()
        return wall_indices

    def _add_random_walls(self, n_walls):
        self.n_random_walls = n_walls
        grid = self.grid
        for _ in range(n_walls):
            empty_locs = torch.nonzero(grid == 0)
            rand_ind = torch.randint(low=0, high=empty_locs.shape[0], size=(1, ))
            new_wall_loc = empty_locs[rand_ind, :].reshape(-1)
            grid[new_wall_loc[0], new_wall_loc[1]] = self._object_map['wall']
            # sample a random free space
            # update the grid with a wall (3)
        self.grid = grid

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
                    torch.tensor([self.width - 1, 1.]),
                    torch.tensor([1., self.height - 1.]),
                    torch.tensor([self.width - 1., self.height - 1.]),
                    self._goal_pos]:
            curr_reward = self._get_reward(agent_pos=pos, goal_pos=self._goal_pos)
            max = curr_reward if curr_reward > max else max
            min = curr_reward if curr_reward < min else min
        self.min_reward = min
        self.max_reward = max

    def get_all_transitions(self, render=False):
        width_range = range(self.width)
        height_range = range(self.height)
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

    def _set_state_for_transitions(self, row, col):
        self.grid[self._goal_pos[0], self._goal_pos[1]] = self._object_map['goal']
        self.grid[self._agent_pos[0], self._agent_pos[1]] = self._object_map['space']
        self.grid[row, col] = self._object_map['agent']
        self._agent_pos = torch.tensor([row, col], dtype=torch.int, device=self.device, requires_grad=False)

