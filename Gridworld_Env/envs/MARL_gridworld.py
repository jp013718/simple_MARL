import numpy as np
import pygame

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

class MARLGridworld(MultiAgentEnv):
  metadata = {"render_modes":["human","rgb_array"], "render_fps": 4}

  def __init__(self, render_mode=None, size=5, max_steps=None):
    super().__init__()
    self.size = size # Size of the gridworld
    self.window_size = 512 # Size of the pygame window

    self._agent_ids = ["agent_1", "agent_2"]

    # Observations for each agent are dictionaries containing their own location and the other agent's location
    # Each location is encoded as a discrete value from ({0, 1, 2, ..., size},{0, 1, 2, ..., size})
    self.observation_space = spaces.Dict(
      {
        "agent_1": spaces.Dict(
          {
            "agent_1": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "agent_2": spaces.Box(0, size - 1, shape=(2,), dtype=int)
          }
        ),
        "agent_2": spaces.Dict(
          {
            "agent_1": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "agent_2": spaces.Box(0, size - 1, shape=(2,), dtype=int)
          }  
        )
      }
    )    

    # We have 4 total actions that can be taken, so we'll represent it with discrete actions
    self.action_space = spaces.Dict(
      {
        "agent_1": spaces.Discrete(4),
        "agent_2": spaces.Discrete(4)
      }
    )

    # This dictionary maps the action space to directions moved for each action
    self._action_to_direction = {
      0: np.array([1,0]), # Right
      1: np.array([0,1]), # Up
      2: np.array([-1,0]), # Left
      3: np.array([0,-1]), # Down
    }

    # Check that the chosen render mode is a legal choice
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    assert max_steps is None or type(max_steps) is int
    self._max_steps = max_steps

    # If human-rendering is used, then window will refer to the window drawn and
    # clock will ensure it is rendered at the correct framerate. Until human render
    # mode is used for the first time, both will remain 'None'
    self.window = None
    self.clock = None

  # Because observations need to be computed both in 'step' and in 'reset', it is sometimes helpful
  # to have a private method to calculate them. However, this is not necessary and can often just
  # as easily be done in the 'step' and 'reset' methods
  def _get_obs(self):
    return {
      "agent_1": {
        "agent_1": self._agent_1_location,
        "agent_2": self._agent_2_location
      },
      "agent_2": {
        "agent_1": self._agent_1_location,
        "agent_2": self._agent_2_location
      }
    }

  # A similar function can be used for the info returned by 'step' and 'reset'. Here, we return the 
  # euclidean distance between the agent and the target
  def _get_info(self):
    return {
      "agent_1": {"current_loc": self._agent_1_location, "distance": np.linalg.norm(self._agent_1_location - self._agent_2_location, ord=1)},
      "agent_2": {"current_loc": self._agent_2_location, "distance": np.linalg.norm(self._agent_1_location - self._agent_2_location, ord=1)}
    }
  
  """
  Reset is called to initiate a new episode. It is safe to assume that 'step' will not be called
  before 'reset'. Additionally, 'reset' should be called whenever a done signal is sent. 'Seed' is
  used to initialize a random number generator. 'Self.np_random' provided by the base class 'gym.env'
  is recommended as the RNG engine. If this is done, there doesn't need to be much worry over seeding,
  but 'super().reset(seed=seed)' must be called to make sure the RNG is correctly seeded. Here, we'll
  randomly set the agent's position and then randomly set the target position until it isn't the same
  as the agent's position.
  """

  # 'Reset' should return a tuple of the initial observation and some additional information. We'll use
  # '_get_obs' and '_get_info' defined earlier.
  def reset(self, seed=None, options=None):
    # Seeding self.np_random
    super().reset(seed=seed)
    self.steps = 0
    # Choose the agent's location at random
    self._agent_1_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    # Sample the target's location at random until it is not the same as the agent's
    self._agent_2_location = self._agent_1_location
    while np.array_equal(self._agent_2_location, self._agent_1_location):
      self._agent_2_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info

  """
  Step usually contains all the logic for the environment. It should accept an action, compute the new
  state of the environment, and then return the 5-tuple containing (observation, reward, terminated, 
  truncated, info). Once the new state has been computed, we check whether the state is terminal. If so,
  'done' should be set accordingly. Since sparse rewards are being used for this gridworld (0 for all
  non-terminal states, 1 for a terminal state), calculating reward is trivial once 'done' is known. Once
  again, we'll use '_get_obs' and '_get_info' for the observations and info.
  """

  def step(self, action):
    # Use the previously created map to interpret the action
    direction_1 = self._action_to_direction[action["agent_1"]]
    direction_2 = self._action_to_direction[action["agent_2"]]
    
    # We can use np.clip to make sure we don't leave the grid. This function makes sure that the first
    # argument falls between the second two
    self._agent_1_location = np.clip(
      self._agent_1_location + direction_1, 0, self.size - 1
    )
    self._agent_2_location = np.clip(
      self._agent_2_location + direction_2, 0, self.size -1
    )

    # An episode is done iff the agent has reached the target
    terminated = {
      "agent_1": np.array_equal(self._agent_1_location, self._agent_2_location),
      "agent_2": np.array_equal(self._agent_1_location, self._agent_2_location),
      "__all__": np.array_equal(self._agent_1_location, self._agent_2_location)
    }

    reward = {
      "agent_1": 50 if terminated["__all__"] else -1,
      "agent_2": 50 if terminated["__all__"] else -1
    }
    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    self.steps += 1
    truncated = {
      "agent_1": False if self._max_steps is None else self.steps >= self._max_steps,
      "agent_2": False if self._max_steps is None else self.steps >= self._max_steps,
      "__all__": False if self._max_steps is None else self.steps >= self._max_steps
    }
    return observation, reward, terminated, truncated, info
  
  """
  Below is an example of using pygame to render the environment. This is used in many gym environments
  and can be used as a skeleton for rendering other environments
  """
  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()
    
  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      # Create a square window
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Define what will be displayed
    canvas = pygame.Surface((self.window_size, self.window_size))
    # Paint a white background
    canvas.fill((255, 255, 255))
    # Define the size of a single grid square in pixels
    pix_square_size = (
      self.window_size / self.size
    )

    # First, draw agent 1
    pygame.draw.circle(
      # Object to draw on
      canvas,
      # Color
      (0, 0, 255),
      # Center
      (self._agent_1_location + 0.5) * pix_square_size,
      # Size
      pix_square_size / 3,
    )

    # Then, draw agent 2
    pygame.draw.circle(
      # Object to draw on
      canvas, 
      # Color
      (255, 0, 0),
      # Center
      (self._agent_2_location + 0.5) * pix_square_size,
      # Size
      pix_square_size / 3,
    )

    # Lastly, add gridlines
    for x in range(self.size + 1):
      pygame.draw.line(
        # Object to draw on
        canvas,
        # Color (black)
        0,
        # Start point
        (0, pix_square_size * x),
        # End point
        (self.window_size, pix_square_size * x),
        # Line weight
        width=3,
      )

      pygame.draw.line(
        canvas,
        0,
        (pix_square_size * x, 0),
        (pix_square_size * x, self.window_size),
        width=3,
      )

    if self.render_mode == "human":
      # Copy the canvas to the window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # Make sure the game is rendered at the chosen fps
      self.clock.tick(self.metadata["render_fps"])
    else: 
      # If not rendering in human mode, return a numpy array representing the game window
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
      )
    
  # Close out any resources held open by the environment. Usually, this is only necessary if pygame
  # is being used to render the window. It may also close files or release other resources being used.
  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()