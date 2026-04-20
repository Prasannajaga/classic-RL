from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ReplayBatch:
    """A minibatch of transitions sampled from replay memory."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminateds: np.ndarray
    truncateds: np.ndarray


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy DQN updates."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.position = 0
        self.size = 0

        self.observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.terminateds = np.empty(capacity, dtype=np.float32)
        self.truncateds = np.empty(capacity, dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition, overwriting the oldest data when full."""

        self.observations[self.position] = np.asarray(observation, dtype=np.float32)
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = np.asarray(next_observation, dtype=np.float32)
        self.terminateds[self.position] = float(terminated)
        self.truncateds[self.position] = float(truncated)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample a random minibatch to decorrelate training updates."""

        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample batch of size {batch_size} from buffer of size {self.size}."
            )

        # Random replay breaks the strong temporal correlations present in
        # sequential experience and makes DQN updates much more stable.
        indices = np.random.randint(0, self.size, size=batch_size)

        return ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminateds=self.terminateds[indices],
            truncateds=self.truncateds[indices],
        )

    def __len__(self) -> int:
        return self.size

