# PPO MiniGrid From Scratch

This project implements Proximal Policy Optimization (PPO) from scratch in PyTorch for MiniGrid. The default task is `MiniGrid-Empty-5x5-v0`, a small grid-world where the agent must navigate to the goal.

## What PPO Is

PPO is an on-policy policy-gradient algorithm. It collects fresh rollouts with the current policy, estimates advantages, then updates the policy while preventing the new policy from moving too far away from the rollout policy.

The policy ratio is:

```text
r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
```

The clipped PPO objective is:

```text
L_clip = E[min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)]
```

In this code, PPO minimizes `actor_loss`, so the clipped objective is negated.

## Environment

The environment is created with:

```python
gymnasium.make(env_id)
```

Stage 1 uses MiniGrid `ImgObsWrapper`, so mission text is ignored. The image observation is flattened into a `float32` vector and normalized by dividing by `255`.

The default environment is:

```text
MiniGrid-Empty-5x5-v0
```

You can later try larger empty rooms such as `MiniGrid-Empty-8x8-v0`.

## Model

`models.py` defines an `ActorCritic` network:

```text
obs -> Linear(obs_dim, 128) -> Tanh
    -> Linear(128, 128) -> Tanh
```

It has two heads:

```text
actor:  Linear(128, n_actions) -> categorical logits
critic: Linear(128, 1)         -> V(s)
```

The actor uses `torch.distributions.Categorical`. During training, actions are sampled. During evaluation, the greedy action is selected with `argmax(logits)`.

## GAE

The rollout buffer computes Generalized Advantage Estimation:

```text
delta_t = reward_t + gamma * next_value * next_non_terminal - value_t

advantage_t = delta_t
            + gamma * gae_lambda * next_non_terminal * advantage_{t+1}
```

Returns are:

```text
R_t = advantage_t + value_t
```

Terminal episodes stop bootstrapping. Time-limit truncations are handled in training by bootstrapping the reward from the final truncated observation before resetting the environment.

## Losses

Critic loss:

```text
L_v = 0.5 * (V(s_t) - R_t)^2
```

Total minimized loss:

```text
L = actor_loss + value_coef * critic_loss - entropy_coef * entropy
```

The entropy bonus encourages exploration. The implementation also logs approximate KL:

```text
approx_kl = mean((ratio - 1) - log_ratio)
```

and clip fraction:

```text
clip_fraction = mean(abs(ratio - 1) > clip_coef)
```

## Install

From this directory:

```bash
pip install -r requirements.txt
```

If you are using the repo venv from the project root:

```bash
../.venv/bin/pip install -r requirements.txt
```

## Train

```bash
python train.py --env_id MiniGrid-Empty-5x5-v0 --total_timesteps 200000
```

From the repo root:

```bash
cd ppo_minigrid_from_scratch
../.venv/bin/python train.py --env_id MiniGrid-Empty-5x5-v0 --total_timesteps 200000
```

Outputs are written to `outputs/` by default:

- `ppo_minigrid_latest.pt`
- `ppo_minigrid_final.pt`
- `metrics.json`
- PNG plots for train rewards, losses, entropy, KL, clip fraction, eval reward, and eval success rate

## Evaluate

```bash
python eval.py --checkpoint outputs/ppo_minigrid_final.pt --env_id MiniGrid-Empty-5x5-v0 --episodes 50
```

## Metrics To Watch

- `train_reward_20`: recent average training reward. It should trend upward.
- `eval_reward`: mean greedy-policy reward. This is the clearest success signal.
- `eval_success`: fraction of eval episodes with reward greater than zero.
- `entropy`: should usually decrease as the policy becomes more confident.
- `approx_kl`: very large values can mean the policy update is too aggressive.
- `clip_fraction`: high values mean many updates are hitting the PPO clipping boundary.
- `critic_loss`: noisy is normal, but exploding values usually indicate unstable value learning.

## Common Debugging Issues

- If `minigrid` is missing, install the project requirements.
- If reward stays flat, try training longer, lowering the learning rate, or increasing `entropy_coef`.
- If `approx_kl` spikes, reduce `learning_rate`, reduce `update_epochs`, or reduce `clip_coef`.
- If `clip_fraction` is almost always zero, the policy update may be too small.
- If `clip_fraction` is very high, updates may be too large.
- If evaluation performs worse than training samples, remember eval is greedy while training samples actions.
- For larger MiniGrid tasks, a flat MLP may become limiting; a CNN encoder is a natural next step.
