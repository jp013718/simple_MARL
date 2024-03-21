import numpy as np
from typing import Dict, Tuple

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class AccCallback(DefaultCallbacks):
  def on_episode_start(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
  ):
    assert episode.length <= 0, (
      "ERROR: 'on_episode_start()' callback should called right "
      "after env reset!"
    )
    # Create values to indicate whether the target has been found
    episode.user_data["target_found"] = 0
    episode.hist_data["target_found"] = 0

  def on_episode_step(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
  ):
    assert episode.length > 0, (
      "ERROR: 'on_episode_step()' callback should not be called right "
      "after env reset!"
    )
    target_found = [1] if np.array_equal(episode.last_info_for("agent_1")["current_loc"], episode.last_info_for("agent_2")["current_loc"]) else [0]
    episode.user_data["target_found"] = target_found

  def on_episode_end(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
  ):
    target_found = episode.user_data["target_found"]
    episode.custom_metrics["target_found"] = target_found
    episode.hist_data["target_found"] = episode.user_data["target_found"]

