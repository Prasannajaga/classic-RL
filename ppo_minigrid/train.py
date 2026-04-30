"""Train PPO from scratch on MiniGrid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from buffer import RolloutBuffer
from config import PPOConfig
from envs import make_env, reset_env, step_env
from eval import evaluate_policy
from models import ActorCritic
from ppo import ppo_update
from utils import plot_metrics, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid.")
    parser.add_argument("--env_id", type=str, default=PPOConfig.env_id)
    parser.add_argument("--total_timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--save_dir", type=str, default=PPOConfig.save_dir)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    global_step: int,
    obs_dim: int,
    n_actions: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "env_id": config.env_id,
            "global_step": global_step,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
        },
        path,
    )


def run_training(config: PPOConfig) -> None:
    set_seed(config.seed)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config.env_id, seed=config.seed)
    obs = reset_env(env, config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    buffer = RolloutBuffer()

    metrics: dict[str, list[float] | list[int]] = {
        "train_steps": [],
        "train_episode_rewards": [],
        "train_episode_lengths": [],
        "update_steps": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "eval_steps": [],
        "eval_reward": [],
        "eval_success_rate": [],
    }

    global_step = 0
    next_eval_step = config.eval_every
    episode_reward = 0.0
    episode_length = 0
    recent_train_rewards: list[float] = []
    last_update_stats = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }

    while global_step < config.total_timesteps:
        buffer.reset()
        rollout_len = min(config.rollout_steps, config.total_timesteps - global_step)

        for _step in range(rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                action_t, logprob_t, _entropy_t, value_t = model.get_action_and_value(obs_t)

            action = int(action_t.item())
            next_obs, reward, terminated, truncated, _info = step_env(env, action)

            done = terminated or truncated
            buffer_reward = reward
            buffer_done = done

            if truncated and not terminated:
                with torch.inference_mode():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _logits, timeout_value = model(next_obs_t)
                    buffer_reward += config.gamma * float(timeout_value.squeeze(-1).item())

            buffer.add(
                obs=obs,
                action=action,
                logprob=float(logprob_t.item()),
                reward=buffer_reward,
                done=buffer_done,
                value=float(value_t.item()),
            )

            global_step += 1
            episode_reward += reward
            episode_length += 1

            if done:
                metrics["train_steps"].append(global_step)
                metrics["train_episode_rewards"].append(float(episode_reward))
                metrics["train_episode_lengths"].append(int(episode_length))
                recent_train_rewards.append(float(episode_reward))
                recent_train_rewards = recent_train_rewards[-20:]
                obs = reset_env(env)
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs

            if global_step >= config.total_timesteps:
                break

        with torch.inference_mode():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _logits, last_value_t = model(last_obs_t)
            last_value = float(last_value_t.squeeze(-1).item())

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        tensors = buffer.get_tensors(returns, advantages, device=device)
        last_update_stats = ppo_update(
            model=model,
            optimizer=optimizer,
            obs=tensors[0],
            actions=tensors[1],
            old_logprobs=tensors[2],
            returns=tensors[3],
            advantages=tensors[4],
            update_epochs=config.update_epochs,
            minibatch_size=config.minibatch_size,
            clip_coef=config.clip_coef,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
        )

        metrics["update_steps"].append(global_step)
        for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"):
            metrics[key].append(float(last_update_stats[key]))

        if global_step >= next_eval_step or global_step >= config.total_timesteps:
            checkpoint_path = save_dir / "ppo_minigrid_latest.pt"
            save_checkpoint(checkpoint_path, model, optimizer, config, global_step, obs_dim, n_actions)
            eval_results = evaluate_policy(
                model=model,
                env_id=config.env_id,
                episodes=config.eval_episodes,
                seed=config.seed + 10_000,
                device=device,
            )
            metrics["eval_steps"].append(global_step)
            metrics["eval_reward"].append(eval_results["mean_reward"])
            metrics["eval_success_rate"].append(eval_results["success_rate"])

            avg_recent_reward = float(np.mean(recent_train_rewards)) if recent_train_rewards else 0.0
            print(
                f"step={global_step} "
                f"train_reward_20={avg_recent_reward:.3f} "
                f"eval_reward={eval_results['mean_reward']:.3f} "
                f"eval_success={eval_results['success_rate']:.3f} "
                f"actor_loss={last_update_stats['actor_loss']:.4f} "
                f"critic_loss={last_update_stats['critic_loss']:.4f} "
                f"entropy={last_update_stats['entropy']:.4f} "
                f"approx_kl={last_update_stats['approx_kl']:.5f} "
                f"clip_fraction={last_update_stats['clip_fraction']:.3f}"
            )

            save_json(save_dir / "metrics.json", metrics)
            plot_metrics(metrics, save_dir)
            next_eval_step += config.eval_every

    final_path = save_dir / "ppo_minigrid_final.pt"
    save_checkpoint(final_path, model, optimizer, config, global_step, obs_dim, n_actions)
    save_json(save_dir / "metrics.json", metrics)
    plot_metrics(metrics, save_dir)
    env.close()


def main() -> None:
    args = parse_args()
    config = PPOConfig(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    run_training(config)


if __name__ == "__main__":
    main()
