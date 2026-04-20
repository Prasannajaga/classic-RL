"""CliffWalking environment for tabular reinforcement learning."""

from __future__ import annotations

from typing import Final


Position = tuple[int, int]


class CliffWalkingEnv:
    """Classic 4x12 CliffWalking environment.

    The grid is indexed row-major from the top-left corner:
    ``state = row * ncols + col``.
    """

    UP: Final[int] = 0
    RIGHT: Final[int] = 1
    DOWN: Final[int] = 2
    LEFT: Final[int] = 3

    def __init__(self, nrows: int = 4, ncols: int = 12) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.n_states = nrows * ncols
        self.n_actions = 4
        self.start: Position = (nrows - 1, 0)
        self.goal: Position = (nrows - 1, ncols - 1)
        self.cliff_cells: set[Position] = {
            (nrows - 1, col) for col in range(1, ncols - 1)
        }
        self.agent_pos: Position = self.start

    def reset(self) -> int:
        """Reset the agent to the start state and return that state index."""
        self.agent_pos = self.start
        return self.pos_to_state(self.agent_pos)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, bool]]:
        """Apply an action and return ``(next_state, reward, done, info)``."""
        if action not in (self.UP, self.RIGHT, self.DOWN, self.LEFT):
            raise ValueError(f"Invalid action: {action}")

        row, col = self.agent_pos
        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.RIGHT:
            col = min(self.ncols - 1, col + 1)
        elif action == self.DOWN:
            row = min(self.nrows - 1, row + 1)
        else:
            col = max(0, col - 1)

        next_pos = (row, col)
        fell = self.is_cliff(next_pos)
        success = next_pos == self.goal

        # negative reward if it fell wrong root and reset and 
        # begin from the starting state 
        if fell:
            self.agent_pos = self.start
            return self.pos_to_state(self.start), -100.0, False, {
                "fell": True,
                "success": False,
            }

        # sucess reached the target 
        self.agent_pos = next_pos
        if success:
            return self.pos_to_state(next_pos), 0.0, True, {
                "fell": False,
                "success": True,
            }

        return self.pos_to_state(next_pos), -1.0, False, {
            "fell": False,
            "success": False,
        }

    def pos_to_state(self, pos: Position) -> int:
        """Encode a grid position as a single integer state."""
        row, col = pos
        return row * self.ncols + col

    def state_to_pos(self, state: int) -> Position:
        """Decode a state index back to a grid position."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"State {state} is out of bounds.")
        return divmod(state, self.ncols)

    def is_cliff(self, pos: Position) -> bool:
        """Return whether a grid position is part of the cliff."""
        return pos in self.cliff_cells
