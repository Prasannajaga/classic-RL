"""Evaluate a saved CliffWalking Q-table with a greedy policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import CliffWalkingEnv
from utils import (
    format_board_grid,
    format_policy_grid,
    greedy_action_from_table,
    summarize_metrics,
    validate_q_table_shape,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-table", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    """Run greedy evaluation episodes and print summary metrics."""
    env = CliffWalkingEnv()
    q_table = np.load(args.q_table)
    validate_q_table_shape(q_table, env)

    metrics: list[dict[str, float | int | bool]] = []
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        cliff_falls = 0
        success = False

        for _ in range(args.max_steps):
            action = greedy_action_from_table(q_table, state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            cliff_falls += int(info["fell"])
            success = bool(info["success"])
            state = next_state

            if done:
                break

        metrics.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "cliff_falls": cliff_falls,
                "success": success,
            }
        )

    summary = summarize_metrics(metrics)
    print(f"Average reward: {summary['average_reward']:.2f}")
    print(f"Average steps: {summary['average_steps']:.2f}")
    print(f"Average falls: {summary['average_falls']:.2f}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print()
    print("Board layout:")
    print(format_board_grid(env))
    print()
    print("Greedy policy:")
    print(format_policy_grid(q_table, env))


def main() -> None:
    """Entrypoint for command-line evaluation."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
