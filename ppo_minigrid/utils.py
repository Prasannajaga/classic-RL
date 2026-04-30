"""General utilities for PPO MiniGrid runs."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Save JSON data with stable indentation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def moving_average(values: list[float], window: int) -> list[float]:
    """Return a simple moving average with the same length as the input."""

    if not values:
        return []
    window = max(1, int(window))
    result = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result.append(float(np.mean(values[start : idx + 1])))
    return result


def _plot_series(
    x: list[int] | list[float],
    y: list[float],
    title: str,
    ylabel: str,
    path: Path,
    smooth: bool = False,
) -> None:
    if not y:
        return

    plt.figure(figsize=(8, 4.5))
    plot_y = moving_average(y, 20) if smooth else y
    plt.plot(x[: len(plot_y)], plot_y)
    plt.title(title)
    plt.xlabel("global step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics(metrics: dict[str, Any], out_dir: str | os.PathLike[str]) -> None:
    """Write training and evaluation plots to disk."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_steps = metrics.get("train_steps", [])
    update_steps = metrics.get("update_steps", [])
    eval_steps = metrics.get("eval_steps", [])

    _plot_series(train_steps, metrics.get("train_episode_rewards", []), "Train Episode Reward", "reward", out_path / "train_episode_reward.png", smooth=True)
    _plot_series(train_steps, metrics.get("train_episode_lengths", []), "Train Episode Length", "length", out_path / "train_episode_length.png", smooth=True)
    _plot_series(update_steps, metrics.get("actor_loss", []), "Actor Loss", "loss", out_path / "actor_loss.png")
    _plot_series(update_steps, metrics.get("critic_loss", []), "Critic Loss", "loss", out_path / "critic_loss.png")
    _plot_series(update_steps, metrics.get("entropy", []), "Entropy", "entropy", out_path / "entropy.png")
    _plot_series(update_steps, metrics.get("approx_kl", []), "Approximate KL", "KL", out_path / "approx_kl.png")
    _plot_series(update_steps, metrics.get("clip_fraction", []), "Clip Fraction", "fraction", out_path / "clip_fraction.png")
    _plot_series(eval_steps, metrics.get("eval_reward", []), "Eval Reward", "reward", out_path / "eval_reward.png")
    _plot_series(eval_steps, metrics.get("eval_success_rate", []), "Eval Success Rate", "success rate", out_path / "eval_success_rate.png")
