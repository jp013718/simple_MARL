from gymnasium.envs.registration import register

register(
  "MARL_Gridworld-v0",
  entry_point="Gridworld_Env.envs:MARLGridworld",
)