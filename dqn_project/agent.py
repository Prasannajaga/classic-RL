from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from dqn_project.configs import DQNConfig
from dqn_project.model import QNetwork
from dqn_project.replay_buffer import ReplayBuffer
from dqn_project.utils import resolve_device


class DQNAgent:
    """Vanilla DQN agent with replay, target network, and checkpointing."""

    def __init__(self, observation_dim: int, num_actions: int, config: DQNConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.observation_dim = observation_dim
        self.num_actions = num_actions

        self.q_network = QNetwork(
            input_dim=observation_dim,
            num_actions=num_actions,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_network = QNetwork(
            input_dim=observation_dim,
            num_actions=num_actions,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_size,
            obs_shape=(observation_dim,),
        )

        self.total_env_steps = 0
        self.total_gradient_steps = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for linear epsilon-greedy exploration."""

        fraction = min(1.0, self.total_env_steps / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Choose an action with epsilon-greedy exploration."""

        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition and advance the environment step counter."""

        self.replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
        )
        self.total_env_steps += 1

    def train_step(self) -> float | None:
        """Run one DQN optimization step if the replay buffer is ready."""

        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        if self.total_env_steps % self.config.train_freq != 0:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch.next_observations,
            dtype=torch.float32,
            device=self.device,
        )
        terminateds = torch.as_tensor(
            batch.terminateds,
            dtype=torch.float32,
            device=self.device,
        )

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._compute_next_q_values(next_states)
            # Only true environment termination should stop bootstrapping.
            # Time-limit truncations still keep the Bellman target's future term.
            targets = rewards + self.config.gamma * (1.0 - terminateds) * next_q_values

        loss = self.loss_fn(current_q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.total_gradient_steps += 1

        if self.total_env_steps % self.config.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Hard-copy online network weights into the target network."""

        # Fixed target updates keep the bootstrap target stable for a while
        # before replacing it with the latest online network parameters.
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute the bootstrap value used in the Bellman target."""

        # This helper keeps vanilla DQN logic isolated, which makes it easy to
        # swap in Double DQN action selection later.
        return self.target_network(next_states).max(dim=1).values

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        """Save model, optimizer, config, and training counters."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "config": self.config.to_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_env_steps": self.total_env_steps,
            "total_gradient_steps": self.total_gradient_steps,
            "epsilon": self.epsilon,
        }
        if extra_state is not None:
            payload["extra_state"] = extra_state

        torch.save(payload, checkpoint_path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint into the current agent instance."""

        checkpoint = torch.load(Path(path), map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
        self.total_gradient_steps = int(checkpoint.get("total_gradient_steps", 0))
        return checkpoint

