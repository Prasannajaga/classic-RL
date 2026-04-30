"""Evaluate a trained PPO MiniGrid checkpoint."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from envs import make_env, reset_env, step_env
from models import ActorCritic
from utils import set_seed


def evaluate_policy(
    model: ActorCritic,
    env_id: str,
    episodes: int = 20,
    seed: int = 42,
    device: torch.device | str = "cpu",
    render_mode: str | None = None,
) -> dict[str, float]:
    """Run greedy-policy evaluation for an in-memory model."""

    env = make_env(env_id, seed=seed, render_mode=render_mode)
    was_training = model.training
    model.eval()

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []

    for episode in range(episodes):
        obs = reset_env(env, seed + episode)
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                logits, _value = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, _info = step_env(env, action)
            done = terminated or truncated
            total_reward += reward
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        successes.append(float(total_reward > 0.0))

    env.close()
    if was_training:
        model.train()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(successes)),
    }


def evaluate(
    checkpoint: str,
    env_id: str | None = None,
    episodes: int = 20,
    seed: int = 42,
    render_mode: str | None = None,
) -> dict[str, float]:
    """Load a checkpoint and run greedy-policy evaluation."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    checkpoint_env_id = checkpoint_data.get("env_id", "MiniGrid-Empty-5x5-v0")
    env_id = env_id or checkpoint_env_id

    env = make_env(env_id, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    env.close()

    model = ActorCritic(obs_dim, n_actions).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    return evaluate_policy(model, env_id, episodes, seed, device, render_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on MiniGrid.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.checkpoint, args.env_id, args.episodes, args.seed, args.render_mode)
    print(f"mean reward: {results['mean_reward']:.3f}")
    print(f"mean episode length: {results['mean_episode_length']:.1f}")
    print(f"success rate: {results['success_rate']:.3f}")


if __name__ == "__main__":
    main()

    
