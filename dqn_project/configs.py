from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(slots=True)
class DQNConfig:
    """Configuration for a vanilla DQN run."""

    env_id: str
    hidden_dims: tuple[int, ...]
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    min_buffer_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_update_freq: int
    max_episodes: int
    train_freq: int = 1
    eval_every_episodes: int = 25
    eval_episodes: int = 5
    gradient_clip_norm: float = 10.0
    seed: int = 42
    device: str = "auto"
    checkpoint_root: str = "output"
    run_name: str | None = None

    def validate(self) -> None:
        """Validate the configuration early so training fails loudly."""

        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in the interval (0, 1].")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be greater than or equal to batch_size.")
        if self.min_buffer_size < self.batch_size:
            raise ValueError("min_buffer_size must be greater than or equal to batch_size.")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError("min_buffer_size cannot exceed buffer_size.")
        if self.epsilon_start < 0.0 or self.epsilon_end < 0.0:
            raise ValueError("epsilon values must be non-negative.")
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("epsilon_start should be greater than or equal to epsilon_end.")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive.")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive.")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive.")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be positive.")
        if self.eval_every_episodes <= 0:
            raise ValueError("eval_every_episodes must be positive.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config for checkpoints and JSON artifacts."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DQNConfig":
        """Reconstruct a config from checkpoint metadata."""

        restored = dict(data)
        restored["hidden_dims"] = tuple(restored["hidden_dims"])
        return cls(**restored)

    def with_overrides(self, **overrides: Any) -> "DQNConfig":
        """Return a new config with CLI overrides applied."""

        sanitized = {key: value for key, value in overrides.items() if value is not None}
        if "hidden_dims" in sanitized:
            sanitized["hidden_dims"] = tuple(sanitized["hidden_dims"])
        return replace(self, **sanitized)


CARTPOLE_CONFIG = DQNConfig(
    env_id="CartPole-v1",
    hidden_dims=(128, 128),
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=100_000,
    min_buffer_size=1_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=20_000,
    target_update_freq=500,
    max_episodes=500,
    eval_every_episodes=25,
    eval_episodes=5,
)

LUNARLANDER_CONFIG = DQNConfig(
    env_id="LunarLander-v3",
    hidden_dims=(256, 256),
    gamma=0.99,
    lr=5e-4,
    batch_size=128,
    buffer_size=200_000,
    min_buffer_size=5_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=100_000,
    target_update_freq=1_000,
    max_episodes=1_500,
    eval_every_episodes=50,
    eval_episodes=10,
)

SIMPLE_CARTPOLE_CONFIG = DQNConfig(
    env_id="SimpleCartPole-v0",
    hidden_dims=(64, 64),
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=50_000,
    min_buffer_size=500,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=10_000,
    target_update_freq=250,
    max_episodes=300,
    eval_every_episodes=25,
    eval_episodes=5,
)

SIMPLE_LUNARLANDER_CONFIG = DQNConfig(
    env_id="SimpleLunarLander-v0",
    hidden_dims=(128, 128),
    gamma=0.99,
    lr=5e-4,
    batch_size=128,
    buffer_size=100_000,
    min_buffer_size=1_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50_000,
    target_update_freq=500,
    max_episodes=800,
    eval_every_episodes=50,
    eval_episodes=10,
)

DEFAULT_CONFIGS: dict[str, DQNConfig] = {
    CARTPOLE_CONFIG.env_id: CARTPOLE_CONFIG,
    LUNARLANDER_CONFIG.env_id: LUNARLANDER_CONFIG,
    SIMPLE_CARTPOLE_CONFIG.env_id: SIMPLE_CARTPOLE_CONFIG,
    SIMPLE_LUNARLANDER_CONFIG.env_id: SIMPLE_LUNARLANDER_CONFIG,
}


def get_config(env_id: str) -> DQNConfig:
    """Return a copy of the default config for the requested environment."""

    if env_id not in DEFAULT_CONFIGS:
        supported = ", ".join(sorted(DEFAULT_CONFIGS))
        raise ValueError(f"Unsupported env_id '{env_id}'. Supported envs: {supported}.")
    return DEFAULT_CONFIGS[env_id].with_overrides()


def supported_env_ids() -> list[str]:
    """List the environments wired into the project."""

    return sorted(DEFAULT_CONFIGS)
