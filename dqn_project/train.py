from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from dqn_project.agent import DQNAgent
from dqn_project.configs import DQNConfig, get_config, supported_env_ids
from dqn_project.utils import (
    ensure_dir,
    evaluate_agent,
    get_env_dimensions,
    make_env,
    plot_training_curves,
    save_json,
    set_seed,
    slugify_env_id,
    timestamp_string,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a DQN agent with PyTorch and Gymnasium.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", choices=supported_env_ids())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--eval-every-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--min-buffer-size", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--target-update-freq", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DQNConfig:
    """Build a training config from defaults plus CLI overrides."""

    config = get_config(args.env_id).with_overrides(
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        checkpoint_root=args.checkpoint_root,
        max_episodes=args.max_episodes,
        eval_every_episodes=args.eval_every_episodes,
        eval_episodes=args.eval_episodes,
        train_freq=args.train_freq,
        hidden_dims=args.hidden_dims,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    config.validate()
    return config


def create_run_dir(config: DQNConfig) -> Path:
    """Create and return the directory for training artifacts."""

    run_name = config.run_name or timestamp_string()
    run_dir = Path(config.checkpoint_root) / slugify_env_id(config.env_id) / run_name
    return ensure_dir(run_dir)


def format_loss(loss: float | None) -> str:
    """Format a loss value for console logging."""

    return "n/a" if loss is None else f"{loss:.4f}"


def run_training(config: DQNConfig) -> Path:
    """Train DQN, log metrics, and save checkpoints."""

    set_seed(config.seed)
    run_dir = create_run_dir(config)
    save_json(config.to_dict(), run_dir / "config.json")

    env = make_env(env_id=config.env_id, seed=config.seed)
    observation_dim, num_actions = get_env_dimensions(env)
    agent = DQNAgent(observation_dim=observation_dim, num_actions=num_actions, config=config)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    best_eval_reward = float("-inf")
    latest_checkpoint_path = run_dir / "latest.pt"
    best_checkpoint_path = run_dir / "best.pt"

    try:
        for episode in range(1, config.max_episodes + 1):
            observation, _ = env.reset(seed=config.seed + episode - 1)
            episode_reward = 0.0
            episode_length = 0
            episode_losses: list[float] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)

                agent.store(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )

                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                observation = next_observation
                episode_reward += float(reward)
                episode_length += 1

            mean_loss = float(np.mean(episode_losses)) if episode_losses else None
            recent_rewards.append(episode_reward)
            moving_average = float(np.mean(recent_rewards))

            row = {
                "episode": episode,
                "episode_reward": round(episode_reward, 6),
                "episode_length": episode_length,
                "epsilon": round(agent.epsilon, 6),
                "mean_loss": "" if mean_loss is None else round(mean_loss, 6),
                "buffer_size": len(agent.replay_buffer),
                "total_env_steps": agent.total_env_steps,
                "total_gradient_steps": agent.total_gradient_steps,
                "terminated": terminated,
                "truncated": truncated,
            }
            train_rows.append(row)
            write_csv(train_rows, run_dir / "train_metrics.csv")

            eval_mean_reward: float | None = None
            if episode % config.eval_every_episodes == 0:
                eval_stats = evaluate_agent(
                    agent=agent,
                    env_id=config.env_id,
                    num_episodes=config.eval_episodes,
                    seed=config.seed + 10_000 + episode,
                )
                eval_mean_reward = float(np.mean([stat.reward for stat in eval_stats]))
                eval_mean_length = float(np.mean([stat.length for stat in eval_stats]))

                eval_row = {
                    "episode": episode,
                    "mean_reward": round(eval_mean_reward, 6),
                    "mean_length": round(eval_mean_length, 6),
                    "num_episodes": config.eval_episodes,
                }
                eval_rows.append(eval_row)
                write_csv(eval_rows, run_dir / "eval_metrics.csv")

                if eval_mean_reward > best_eval_reward:
                    best_eval_reward = eval_mean_reward
                    agent.save(
                        best_checkpoint_path,
                        extra_state={
                            "episode": episode,
                            "best_eval_reward": best_eval_reward,
                            "run_dir": str(run_dir),
                        },
                    )

                agent.save(
                    latest_checkpoint_path,
                    extra_state={
                        "episode": episode,
                        "best_eval_reward": best_eval_reward,
                        "last_eval_reward": eval_mean_reward,
                        "run_dir": str(run_dir),
                    },
                )

            eval_fragment = (
                f" | Eval {eval_mean_reward:7.2f} | Best {best_eval_reward:7.2f}"
                if eval_mean_reward is not None
                else ""
            )
            print(
                f"Episode {episode:4d}/{config.max_episodes} | "
                f"Reward {episode_reward:8.2f} | Avg20 {moving_average:8.2f} | "
                f"Loss {format_loss(mean_loss):>8} | Eps {agent.epsilon:6.3f} | "
                f"Buffer {len(agent.replay_buffer):7d} | Steps {agent.total_env_steps:7d}"
                f"{eval_fragment}"
            )

        if not eval_rows:
            agent.save(
                best_checkpoint_path,
                extra_state={
                    "episode": config.max_episodes,
                    "best_eval_reward": None,
                    "run_dir": str(run_dir),
                },
            )
        agent.save(
            latest_checkpoint_path,
            extra_state={
                "episode": config.max_episodes,
                "best_eval_reward": None if best_eval_reward == float("-inf") else best_eval_reward,
                "run_dir": str(run_dir),
            },
        )

        plot_training_curves(train_rows=train_rows, eval_rows=eval_rows, output_dir=run_dir)
        summary = {
            "env_id": config.env_id,
            "device": str(agent.device),
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
            "episodes": config.max_episodes,
            "total_env_steps": agent.total_env_steps,
            "total_gradient_steps": agent.total_gradient_steps,
            "best_eval_reward": None if best_eval_reward == float("-inf") else best_eval_reward,
            "final_training_reward": train_rows[-1]["episode_reward"] if train_rows else None,
        }
        save_json(summary, run_dir / "summary.json")
    finally:
        env.close()

    return run_dir


def main() -> None:
    """CLI entrypoint for DQN training."""

    args = parse_args()
    config = build_config(args)
    run_dir = run_training(config)
    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()

