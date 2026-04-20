"""Tabular reinforcement learning agents for CliffWalking."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


QTable = npt.NDArray[np.float64]


class TabularAgent(ABC):
    """Common functionality shared by tabular agents."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table: QTable = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Select an action with epsilon-greedy exploration."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action_with_random_tie_break(state)

    def decay_epsilon_value(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _greedy_action_with_random_tie_break(self, state: int) -> int:
        q_values = self.q_table[state]
        max_value = float(np.max(q_values))
        best_actions = np.flatnonzero(np.isclose(q_values, max_value))
        return int(self.rng.choice(best_actions))

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        """Update the Q-table from a transition."""


class SarsaAgent(TabularAgent):
    """On-policy SARSA agent."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        if done:
            target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA requires next_action for non-terminal updates.")
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


class QLearningAgent(TabularAgent):
    """Off-policy Q-learning agent."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        del next_action

        # if terminal hit the final state use that current one here 
        if done:
            target = reward
        # here apply the greedy arg max from Q-learning 
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
