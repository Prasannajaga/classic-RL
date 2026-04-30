"""Actor-critic network used by PPO."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared MLP actor-critic for discrete MiniGrid actions."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and scalar value estimates."""

        hidden = self.network(obs)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action and return PPO training quantities."""

        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        logprob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, logprob, entropy, value.squeeze(-1)
