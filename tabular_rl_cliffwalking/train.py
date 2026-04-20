"""Train SARSA or Q-learning on the CliffWalking environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from agents import QLearningAgent, SarsaAgent, TabularAgent
from env import CliffWalkingEnv
from utils import ensure_dir, plot_metric_curve, save_metrics, summarize_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("sarsa", "qlearning"), required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store q_table.npy, metrics.json, and training curves.",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace, env: CliffWalkingEnv) -> TabularAgent:
    """Create the requested tabular agent."""
    common_kwargs = {
        "n_states": env.n_states,
        "n_actions": env.n_actions,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "min_epsilon": args.min_epsilon,
        "seed": args.seed,
    }
    if args.algo == "sarsa":
        return SarsaAgent(**common_kwargs)
    return QLearningAgent(**common_kwargs)


def train(args: argparse.Namespace) -> Path:
    """Run training and save the resulting artifacts."""
    env = CliffWalkingEnv()
    agent = build_agent(args, env)

    output_dir = args.output_dir or Path("outputs") / args.algo
    ensure_dir(output_dir)

    metrics: list[dict[str, float | int | bool]] = []
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        action = agent.select_action(state)

        total_reward = 0.0
        steps = 0
        cliff_falls = 0
        success = False

        for _ in range(args.max_steps):
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            cliff_falls += int(info["fell"])
            success = bool(info["success"])

            if args.algo == "sarsa":
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, done, next_action)
            else:
                agent.update(state, action, reward, next_state, done)
                next_action = agent.select_action(next_state) if not done else None

            state = next_state
            if done:
                break
            action = int(next_action)

        agent.decay_epsilon_value()
        metrics.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "cliff_falls": cliff_falls,
                "success": success,
            }
        )

        if episode % 500 == 0:
            window_metrics = metrics[-500:]
            window_summary = summarize_metrics(window_metrics)
            print(
                f"Episode {episode}/{args.episodes} | "
                f"avg reward: {window_summary['average_reward']:.2f} | "
                f"success rate: {window_summary['success_rate']:.2%} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    np.save(output_dir / "q_table.npy", agent.q_table)
    save_metrics(metrics, output_dir / "metrics.json", config=serialize_args(args, output_dir))
    plot_metric_curve(
        [float(item["total_reward"]) for item in metrics],
        title=f"{args.algo.title()} reward per episode",
        ylabel="Total reward",
        output_path=output_dir / "reward_curve.png",
    )
    plot_metric_curve(
        [int(item["steps"]) for item in metrics],
        title=f"{args.algo.title()} steps per episode",
        ylabel="Steps",
        output_path=output_dir / "steps_curve.png",
    )
    plot_metric_curve(
        [int(item["cliff_falls"]) for item in metrics],
        title=f"{args.algo.title()} cliff falls per episode",
        ylabel="Falls",
        output_path=output_dir / "falls_curve.png",
    )

    overall = summarize_metrics(metrics)
    print(
        f"Training complete | avg reward: {overall['average_reward']:.2f} | "
        f"success rate: {overall['success_rate']:.2%} | output: {output_dir}"
    )
    return output_dir


def serialize_args(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Convert parsed arguments into JSON-serializable config."""
    return {
        "algo": args.algo,
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "min_epsilon": args.min_epsilon,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }


def main() -> None:
    """Entrypoint for command-line training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
