"""Rollout storage and GAE computation."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """A simple single-environment rollout buffer for PPO."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.logprobs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Append one transition to the buffer."""

        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation and returns."""

        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = float(last_value)
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def get_tensors(
        self,
        returns: np.ndarray,
        advantages: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert rollout arrays to tensors for PPO updates."""

        obs_array = np.stack(self.observations).astype(np.float32, copy=False)
        actions_array = np.asarray(self.actions, dtype=np.int64)
        logprobs_array = np.asarray(self.logprobs, dtype=np.float32)

        obs = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_array, dtype=torch.long, device=device)
        old_logprobs = torch.as_tensor(logprobs_array, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return obs, actions, old_logprobs, returns_t, advantages_t
