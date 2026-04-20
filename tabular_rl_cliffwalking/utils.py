"""Utility helpers for training, evaluation, and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from env import CliffWalkingEnv


QTable = npt.NDArray[np.float64]
MetricRecord = dict[str, float | int | bool]

ACTION_SYMBOLS: dict[int, str] = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_metrics(metrics: Sequence[MetricRecord]) -> dict[str, float]:
    """Compute aggregate statistics over a sequence of episode metrics."""
    if not metrics:
        return {
            "average_reward": 0.0,
            "average_steps": 0.0,
            "average_falls": 0.0,
            "success_rate": 0.0,
        }

    count = len(metrics)
    return {
        "average_reward": float(sum(float(item["total_reward"]) for item in metrics) / count),
        "average_steps": float(sum(int(item["steps"]) for item in metrics) / count),
        "average_falls": float(sum(int(item["cliff_falls"]) for item in metrics) / count),
        "success_rate": float(sum(bool(item["success"]) for item in metrics) / count),
    }


def save_metrics(
    metrics: Sequence[MetricRecord],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Save training metrics and config to a JSON file."""
    payload = {
        "config": config,
        "summary": summarize_metrics(metrics),
        "episodes": list(metrics),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def plot_metric_curve(
    values: Sequence[float | int],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot and save a single metric curve."""
    numeric_values = np.asarray(values, dtype=np.float64)
    plt.figure(figsize=(10, 4))
    plt.plot(numeric_values, linewidth=1.0, alpha=0.75, label="per episode")

    if numeric_values.size >= 10:
        window = min(100, numeric_values.size)
        kernel = np.ones(window, dtype=np.float64) / window
        moving_average = np.convolve(numeric_values, kernel, mode="valid")
        x_values = np.arange(window - 1, numeric_values.size)
        plt.plot(x_values, moving_average, linewidth=2.0, label=f"{window}-episode avg")
        plt.legend()

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def validate_q_table_shape(q_table: QTable, env: CliffWalkingEnv) -> None:
    """Validate that a Q-table matches the environment dimensions."""
    expected_shape = (env.n_states, env.n_actions)
    if q_table.shape != expected_shape:
        raise ValueError(
            f"Expected q_table shape {expected_shape}, but received {q_table.shape}."
        )


def greedy_action_from_table(q_table: QTable, state: int) -> int:
    """Return the greedy action for a state using deterministic tie-breaking."""
    return int(np.argmax(q_table[state]))


def format_policy_grid(q_table: QTable, env: CliffWalkingEnv) -> str:
    """Render the greedy policy as a grid of arrows."""
    validate_q_table_shape(q_table, env)

    rows: list[str] = []
    for row in range(env.nrows):
        cells: list[str] = []
        for col in range(env.ncols):
            pos = (row, col)
            if pos == env.start:
                cells.append("S")
            elif pos == env.goal:
                cells.append("G")
            elif pos in env.cliff_cells:
                cells.append("C")
            else:
                state = env.pos_to_state(pos)
                cells.append(ACTION_SYMBOLS[greedy_action_from_table(q_table, state)])
        rows.append(" ".join(cells))
    return "\n".join(rows)
