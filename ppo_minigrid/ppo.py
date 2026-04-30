"""PPO optimization step."""

from __future__ import annotations

import torch
from torch import nn

from models import ActorCritic


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run PPO clipped-policy optimization over rollout tensors."""

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = obs.shape[0]

    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []

    for _epoch in range(update_epochs):
        indices = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]

            _, new_logprob, entropy, new_value = model.get_action_and_value(
                obs[mb_idx],
                actions[mb_idx],
            )

            # here we normalize old vs new probs 
            log_ratio = new_logprob - old_logprobs[mb_idx]
            ratio = log_ratio.exp()

            # here we use clipped and unclipped version of the PPO 
            minibatch_advantages = advantages[mb_idx]
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * minibatch_advantages
            actor_loss = -torch.min(unclipped, clipped).mean()

            critic_loss = 0.5 * ((new_value - returns[mb_idx]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy_loss.item())
            approx_kls.append(approx_kl.item())
            clip_fractions.append(clip_fraction.item())

    return {
        "actor_loss": sum(actor_losses) / len(actor_losses),
        "critic_loss": sum(critic_losses) / len(critic_losses),
        "entropy": sum(entropies) / len(entropies),
        "approx_kl": sum(approx_kls) / len(approx_kls),
        "clip_fraction": sum(clip_fractions) / len(clip_fractions),
    }
