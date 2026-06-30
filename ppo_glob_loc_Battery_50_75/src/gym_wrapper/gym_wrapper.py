

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MaskableGymWrapper(gym.Wrapper):
    """
    Makes a PathPlanning gym compatible with sb3_contrib.MaskablePPO.

    Key behaviours:
    - action_masks() returns the current boolean mask (shape: [n_actions])
    - "mask" key is removed from observations and observation_space
    - All other obs keys (global_map, local_map, scalars) pass through unchanged
    """

    def __init__(self, env):
        super().__init__(env)
        # Remove "mask" from the observation space so SB3 policy builds correctly
        orig = env.observation_space
        assert isinstance(orig, spaces.Dict), \
            "Expected Dict observation space from PathPlanning gym"

        filtered = {k: v for k, v in orig.spaces.items() if k != "mask"}
        self.observation_space = spaces.Dict(filtered)

        # Cache the last mask so action_masks() can return it between steps
        self._last_mask = np.ones(env.action_space.n, dtype=bool)

    def _filter_obs(self, obs: dict) -> dict:
        """Strip mask from obs dict and cache it."""
        obs = dict(obs)  # shallow copy — don't mutate original
        if "mask" in obs:
            # obs["mask"] shape is (1, n_actions) from observation.py — flatten
            self._last_mask = obs.pop("mask").flatten().astype(bool)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._filter_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._filter_obs(obs)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Called by MaskablePPO at every step to enforce action masking.
        Returns boolean array of shape (n_actions,): True = action allowed.
        """
        return self._last_mask