from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Simple MLP that maps vector observations to action values."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value per action for each observation in the batch."""

        return self.network(observations)

