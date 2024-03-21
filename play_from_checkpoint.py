import os
import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from Gridworld_Env.envs.MARL_gridworld import MARLGridworld

from ray.tune.registry import register_env

register_env("MARL_Gridworld-v0", lambda config: MARLGridworld(max_steps=50))

checkpoint_dir = "results/algo_10"

algos = Algorithm.from_checkpoint(checkpoint_dir)
print(algos)

env = gym.make("MARL_Gridworld-v0", render_mode = "human")
observation, info = env.reset()

for _ in range(3000):
  env.render()
  action = {}
  action["agent_1"] = algos.compute_single_action(observation["agent_1"], policy_id="agent_1")
  action["agent_2"] = algos.compute_single_action(observation["agent_2"], policy_id="agent_2")
  observation, reward, terminated, truncated, info = env.step(action)

  if terminated["__all__"] or truncated["__all__"]:
    observation, info = env.reset()