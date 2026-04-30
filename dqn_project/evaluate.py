from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dqn_project.agent import DQNAgent
from dqn_project.configs import DQNConfig
from dqn_project.utils import evaluate_agent, get_env_dimensions, make_env


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint dictionary without constructing the agent first."""

    return torch.load(checkpoint_path, map_location="cpu")


def main() -> None:
    """CLI entrypoint for greedy policy evaluation."""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint_metadata(checkpoint_path)

    checkpoint_config = DQNConfig.from_dict(checkpoint["config"])
    config = checkpoint_config.with_overrides(
        env_id=args.env_id,
        device=args.device,
    )
    config.validate()

    env = make_env(env_id=config.env_id, seed=args.seed)
    observation_dim, num_actions = get_env_dimensions(env)
    env.close()

    agent = DQNAgent(observation_dim=observation_dim, num_actions=num_actions, config=config)
    agent.load(checkpoint_path)

    results = evaluate_agent(
        agent=agent,
        env_id=config.env_id,
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
    )

    for result in results:
        print(
            f"Eval Episode {result.episode:3d} | Reward {result.reward:8.2f} | "
            f"Length {result.length:4d} | Terminated {result.terminated} | "
            f"Truncated {result.truncated}"
        )

    rewards = [result.reward for result in results]
    lengths = [result.length for result in results]
    print(
        f"Summary | Env {config.env_id} | Episodes {args.episodes} | "
        f"Mean Reward {np.mean(rewards):.2f} | Std Reward {np.std(rewards):.2f} | "
        f"Mean Length {np.mean(lengths):.2f}"
    )

    
if __name__ == "__main__":
    main()

