"""MiniGrid environment helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper


class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten MiniGrid image observations and normalize them to [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        flat_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32).reshape(-1) / 255.0


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create a MiniGrid env with image-only flattened observations."""

    env = gym.make(env_id, render_mode=render_mode)
    env = ImgObsWrapper(env)
    env = FlatObsWrapper(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Reset an env and return only the processed observation."""

    obs, _info = env.reset(seed=seed)
    return obs


def step_env(env: gym.Env, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Step an env using the Gymnasium API."""

    obs, reward, terminated, truncated, info = env.step(action)
    return obs, float(reward), bool(terminated), bool(truncated), info
