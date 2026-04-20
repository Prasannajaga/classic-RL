from __future__ import annotations

import csv
import json
import random
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

if TYPE_CHECKING:
    from dqn_project.agent import DQNAgent


@dataclass(slots=True)
class EpisodeStats:
    """Summary statistics for one environment episode."""

    episode: int
    reward: float
    length: int
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_is_available() -> bool:
    """Check CUDA availability without noisy driver warnings in auto mode."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization:.*")
        return torch.cuda.is_available()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve 'auto', 'cpu', or 'cuda' into an actual torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if cuda_is_available() else "cpu")
    if device_name == "cuda" and not cuda_is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(device_name)


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    """Construct and seed a Gymnasium environment."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="pkg_resources is deprecated as an API.*",
        )
        env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env


def get_env_dimensions(env: gym.Env) -> tuple[int, int]:
    """Validate the environment matches the vector-observation DQN setup."""

    if not isinstance(env.observation_space, Box):
        raise TypeError("This project expects a continuous Box observation space.")
    if env.observation_space.shape is None or len(env.observation_space.shape) != 1:
        raise ValueError("This DQN implementation only supports 1D vector observations.")
    if not isinstance(env.action_space, Discrete):
        raise TypeError("This project expects a discrete action space.")
    return int(env.observation_space.shape[0]), int(env.action_space.n)


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it for convenience."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    """Return a compact timestamp for run folder names."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify_env_id(env_id: str) -> str:
    """Convert an environment id into a filesystem-friendly name."""

    return env_id.lower().replace("/", "_")


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist JSON with stable formatting."""

    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write a list of dictionaries to a CSV file."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    moving_average_window: int | None = None,
) -> None:
    """Save a line plot, optionally overlaying a moving average."""

    if not x_values or not y_values:
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, label="raw", linewidth=1.6, alpha=0.8)

    if moving_average_window is not None and len(y_values) >= moving_average_window:
        kernel = np.ones(moving_average_window, dtype=np.float32) / moving_average_window
        averaged = np.convolve(np.asarray(y_values, dtype=np.float32), kernel, mode="valid")
        averaged_x = x_values[moving_average_window - 1 :]
        plt.plot(averaged_x, averaged, label=f"{moving_average_window}-episode avg", linewidth=2.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if moving_average_window is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_curves(
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create simple plots for the most important training metrics."""

    if train_rows:
        episodes = [int(row["episode"]) for row in train_rows]
        rewards = [float(row["episode_reward"]) for row in train_rows]
        epsilons = [float(row["epsilon"]) for row in train_rows]
        losses = [
            float(row["mean_loss"])
            for row in train_rows
            if row["mean_loss"] not in ("", None)
        ]
        loss_episodes = [
            int(row["episode"])
            for row in train_rows
            if row["mean_loss"] not in ("", None)
        ]

        _plot_curve(
            x_values=episodes,
            y_values=rewards,
            path=output_dir / "reward_curve.png",
            title="Training Reward per Episode",
            xlabel="Episode",
            ylabel="Reward",
            moving_average_window=20,
        )
        _plot_curve(
            x_values=episodes,
            y_values=epsilons,
            path=output_dir / "epsilon_curve.png",
            title="Exploration Schedule",
            xlabel="Episode",
            ylabel="Epsilon",
        )
        _plot_curve(
            x_values=loss_episodes,
            y_values=losses,
            path=output_dir / "loss_curve.png",
            title="Mean Training Loss per Episode",
            xlabel="Episode",
            ylabel="SmoothL1 Loss",
        )

    if eval_rows:
        eval_episodes = [int(row["episode"]) for row in eval_rows]
        eval_rewards = [float(row["mean_reward"]) for row in eval_rows]
        _plot_curve(
            x_values=eval_episodes,
            y_values=eval_rewards,
            path=output_dir / "eval_reward_curve.png",
            title="Greedy Evaluation Reward",
            xlabel="Episode",
            ylabel="Mean Reward",
        )


def evaluate_agent(
    agent: "DQNAgent",
    env_id: str,
    num_episodes: int,
    seed: int,
    render: bool = False,
) -> list[EpisodeStats]:
    """Run greedy evaluation episodes and return per-episode stats."""

    render_mode = "human" if render else None
    env = make_env(env_id=env_id, seed=seed, render_mode=render_mode)

    try:
        results: list[EpisodeStats] = []
        for episode in range(1, num_episodes + 1):
            observation, _ = env.reset(seed=seed + episode - 1)
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation, greedy=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                observation = next_observation

            results.append(
                EpisodeStats(
                    episode=episode,
                    reward=episode_reward,
                    length=episode_length,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
        return results
    finally:
        env.close()
