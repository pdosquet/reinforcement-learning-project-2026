"""Custom Gymnasium wrappers for PyFlyt environments."""

import gymnasium
import numpy as np
from gymnasium import spaces


class WaypointDistanceShaping(gymnasium.RewardWrapper):
    """Potential-based reward shaping for PyFlyt Waypoints envs.

    Adds φ(s') − φ(s) to each step reward, where:
        φ(s) = −shaping_coef × Σ ‖target_delta_i‖  (sum over remaining waypoints)

    Using the sum of all remaining distances eliminates the reward spike at
    waypoint transitions: the completed delta is ≈ 0 just before it drops from
    the list, so the potential changes smoothly.

    Must be applied BEFORE FlattenWaypointEnv (inner wrapper) so it sees the
    raw Dict observation with 'target_deltas'.
    """

    def __init__(self, env, shaping_coef: float = 0.01):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self._prev_potential: float = 0.0

    def _potential(self, obs) -> float:
        deltas = obs["target_deltas"]  # (N, 3)
        return -self.shaping_coef * float(np.linalg.norm(deltas, axis=1).sum())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential(obs)
        return obs, info

    def reward(self, reward):
        # Called by gymnasium after step(); obs already updated via self.env.step()
        # We need the current obs — access via the unwrapped env's last obs.
        # gymnasium.RewardWrapper does not pass obs here, so we cache it in step().
        shaping = self._curr_potential - self._prev_potential
        self._prev_potential = self._curr_potential
        return reward + shaping

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._curr_potential = self._potential(obs)
        shaped_reward = self.reward(reward)
        return obs, shaped_reward, terminated, truncated, info


class FlattenWaypointEnv(gymnasium.ObservationWrapper):
    """Flattens the Dict observation of PyFlyt Waypoints envs into a single Box.

    The Waypoints env returns:
      - 'attitude': Box(21,) - drone state
      - 'target_deltas': (N, 3) waypoint deltas — N can decrease as waypoints are reached

    This wrapper pads/truncates target_deltas to a fixed number of waypoints
    and concatenates everything into a single flat vector.
    """

    def __init__(self, env, max_waypoints=4):
        super().__init__(env)
        self.max_waypoints = max_waypoints

        # Determine attitude dim from the observation space
        self.attitude_dim = env.observation_space["attitude"].shape[0]
        total_dim = self.attitude_dim + self.max_waypoints * 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )

    def observation(self, obs):
        attitude = obs["attitude"]
        targets = obs["target_deltas"]  # shape (N, 3), N may vary

        # Pad or truncate to max_waypoints
        padded = np.zeros((self.max_waypoints, 3), dtype=np.float64)
        n = min(len(targets), self.max_waypoints)
        padded[:n] = targets[:n]

        return np.concatenate([attitude, padded.flatten()])
