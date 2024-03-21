import ray
ray.init(ignore_reinit_error=True, local_mode=True)
import gymnasium as gym

from ray.tune.registry import register_env
from Gridworld_Env.envs.MARL_gridworld import MARLGridworld

register_env("MARL_Gridworld-v0", lambda config: MARLGridworld(max_steps=50))

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from callbacks.acc_callback import AccCallback

temp_env = gym.make("MARL_Gridworld-v0")

config = (
  PPOConfig()
  .environment("MARL_Gridworld-v0")
  .framework("torch")
  .evaluation(evaluation_num_workers=1, evaluation_interval=50) 
  .callbacks(AccCallback)
  .multi_agent(
    policies={
      "agent_1": PolicySpec(None, temp_env.observation_space['agent_1'], temp_env.action_space['agent_1'], {}),
      "agent_2": PolicySpec(None, temp_env.observation_space['agent_2'], temp_env.action_space['agent_2'], {})
    },
    policies_to_train=["agent_1", "agent_2"],
    policy_mapping_fn=(
      lambda agent_id, episode, worker, **kw: ("agent_1" if agent_id == "agent_1" else "agent_2")
    )
  )
)

# Build allgorithm config
algo = config.build()

# Number of episodes
train_duration = 10
elapsed_time = 0
num_episodes = 0

# Training loop
while True:
  result = algo.train()
  print(f'Episode: {num_episodes}')
  print(f'Max Reward: {result["episode_reward_max"]}')
  print(f'Min Reward: {result["episode_reward_min"]}')
  print(f'Mean Reward: {result["episode_reward_mean"]}')
  print(f'Time Spent this Episode: {result["time_total_s"]-elapsed_time}s')
  print(f'Accuracy this Episode: {result["custom_metrics"]["target_found_mean"]}')
  elapsed_time = result["time_total_s"]
  print(f'Elapsed Time: {int(elapsed_time)}s')
  print()
  
  if num_episodes % 2 == 0:
    save_result = algo.save(checkpoint_dir=f"results/algo_{num_episodes}")
    save_path = save_result.checkpoint.path
    print(f"Checkpoint reached. New algorithm saved to {save_path}")
  
  if num_episodes >= train_duration:
    save_result = algo.save(checkpoint_dir=f"results/algo_{num_episodes}")
    save_path = save_result.checkpoint.path
    print(f"Training complete. Final algorithm saved to {save_path}")
    break

  num_episodes += 1
