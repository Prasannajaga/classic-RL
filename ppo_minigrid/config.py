"""Configuration for PPO MiniGrid experiments."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    env_id: str = "MiniGrid-Empty-5x5-v0"
    seed: int = 42
    total_timesteps: int = 200_000
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_every: int = 10_000
    eval_episodes: int = 20
    save_dir: str = "outputs"
