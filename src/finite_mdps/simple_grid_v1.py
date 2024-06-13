import torch
from src.finite_mdps.torch_helpers import TorchBox
import gymnasium as gym
import pygame

class SimpleGridV1(gym.Env):

    def __init__(self, width, height, device, seed=0, render_mode=None, reward_func='manhattan'):
        super().__init__()
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
        assert isinstance(width, int) and isinstance(height, int)
        assert width >= 10 and height >= 10
        assert isinstance(device, torch.device)
        assert isinstance(seed, int)
        assert render_mode in ["human", "rgb_array"]
        assert reward_func in ["manhattan", "euclidean", "sparse"]

        self.width = width
        self.height = height
        self.device = device
        self.seed = seed
        self.render_mode = render_mode
        self.reward_func = reward_func

        self._agent_pos = None
        self._init_agent_pos = None
        self._goal_pos = None
        self._init_goal_pos = None
        self._timestep = 0
        self._max_timestep = 100

        low = torch.zeros(size=(self.width * self.height, ), dtype=torch.int, device=self.device, requires_grad=False)
        high = 3 * torch.ones(size=(self.width * self.height, ), dtype=torch.int, device=self.device, requires_grad=False)
        self.observation_space = TorchBox(low=low, high=high)
        self.action_space = gym.spaces.Discrete(5) # north, east, south, west, nop

        self._action_map = {0: torch.tensor([ 0, -1], dtype=torch.int, device=self.device, requires_grad=False),
                            1: torch.tensor([ 1,  0], dtype=torch.int, device=self.device, requires_grad=False),
                            2: torch.tensor([ 0,  1], dtype=torch.int, device=self.device, requires_grad=False),
                            3: torch.tensor([-1,  0], dtype=torch.int, device=self.device, requires_grad=False)}

        self._object_tensor_map = {0: torch.tensor([0], dtype=torch.int, device=self.device, requires_grad=False),
                                   1: torch.tensor([1], dtype=torch.int, device=self.device, requires_grad=False),
                                   2: torch.tensor([2], dtype=torch.int, device=self.device, requires_grad=False),
                                   3: torch.tensor([3], dtype=torch.int, device=self.device, requires_grad=False)}

        self._reset_rng(seed=self.seed)
        self._initialise()

        if self.reward_func == 'euclidean':
            self._get_reward = self._get_reward_euclidean
            # self._init_reward_scalar = 0
            # self._init_reward_scalar = self._get_reward(agent_pos=self._init_agent_pos, goal_pos=self._init_goal_pos)
        elif self.reward_func == 'manhattan':
            self._get_reward = self._get_reward_manhattan
            # self._init_reward_scalar = 0
            # self._init_reward_scalar = self._get_reward(agent_pos=self._init_agent_pos, goal_pos=self._init_goal_pos)
        elif self.reward_func == 'sparse':
            self._get_reward = self._get_reward_sparse
        else:
            raise NotImplementedError

        self._pygame_window = None
        self._pygame_window_width = None
        self._pygame_window_height = None
        self._pygame_clock = None
        self._pygame_square_pix = 20

    def step(self, action):
        assert self._timestep <= self._max_timestep, 'episode has terminated'
        next_state, reward = self._update_agent_pos(state=self.grid, action=action)

        self.grid = next_state
        observation = self.grid.flatten()

        terminated = False
        truncated = True if self._timestep >= self._max_timestep else False
        info = {}

        self._timestep += 1

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
        grid[self._agent_pos[0], self._agent_pos[1]] = 1
        self._goal_pos = self._init_goal_pos = self._reset_goal_pos()
        grid[self._goal_pos[0], self._goal_pos[1]] = 2
        grid[0, :] = 3
        grid[-1, :] = 3
        grid[:, 0] = 3
        grid[:, -1] = 3
        self.grid = grid

    def _reset_agent_pos(self):
        # agent_pos = torch.tensor([2, 2], dtype=torch.int, device=self.device, requires_grad=False)
        pos_x = torch.randint(low=1, high=self.width-1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
        pos_y = torch.randint(low=1, high=int(self.height/4), size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
        agent_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return agent_pos

    def _reset_goal_pos(self):
        # goal_pos = torch.tensor([self.width-3, self.height-3], dtype=torch.float, device=self.device, requires_grad=False)
        pos_x = torch.randint(low=1, high=self.width - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
        pos_y = torch.randint(low=int(3 * self.height / 4), high=self.height - 1, size=(1, ), dtype=torch.int, device=self.device, requires_grad=False)
        goal_pos = torch.tensor([pos_x, pos_y], dtype=torch.int, device=self.device, requires_grad=False)
        return goal_pos

    def _update_agent_pos(self, state, action):
        curr_agent_pos = self._agent_pos
        action_tensor = self._action_map[action]

        new_agent_pos = curr_agent_pos + action_tensor
        next_state = state.clone().detach()

        if state[new_agent_pos[0], new_agent_pos[1]] == self._object_tensor_map[3]:
            new_agent_pos = curr_agent_pos
        else:
            next_state[curr_agent_pos[0], curr_agent_pos[1]] = 0
            next_state[new_agent_pos[0], new_agent_pos[1]] = 1

        self._agent_pos = new_agent_pos

        reward = self._get_reward(agent_pos=new_agent_pos, goal_pos = self._goal_pos)
        return next_state, reward

    def _get_walls(self):
        grid = self.grid
        walls = (grid == 3)
        wall_indices = walls.nonzero().tolist()
        return wall_indices

    def _get_reward_manhattan(self, **kwargs):
        assert kwargs['agent_pos'] is not None
        assert kwargs['goal_pos'] is not None
        agent_pos = kwargs['agent_pos']
        goal_pos = kwargs['goal_pos']
        reward = -1.0 * torch.sum(torch.abs((agent_pos - goal_pos)).to(torch.float)) # - self._init_reward_scalar
        return reward

    def _get_reward_euclidean(self, **kwargs):
        assert kwargs['agent_pos'] is not None
        assert kwargs['goal_pos'] is not None
        agent_pos = kwargs['agent_pos']
        goal_pos = kwargs['goal_pos']
        reward = -1.0 * torch.linalg.norm((agent_pos - goal_pos).to(torch.float)) # - self._init_reward_scalar
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



