# DQN Project

This project implements vanilla Deep Q-Networks from scratch in Python using PyTorch and Gymnasium. It is designed to be educational enough to study line by line, but organized cleanly enough to reuse for multiple control environments.

## Files

- `configs.py`: default hyperparameters for CartPole and LunarLander
- `replay_buffer.py`: random replay memory for off-policy updates
- `model.py`: MLP Q-network for vector observations
- `agent.py`: DQN agent, Bellman update, target network, checkpointing
- `train.py`: full training loop with evaluation, logging, and checkpoint saving
- `evaluate.py`: greedy policy evaluation script
- `utils.py`: seeding, plotting, env helpers, CSV/JSON output

## How DQN Works

DQN replaces the tabular Q-function with a neural network `Q(s, a)` that predicts one value per action from the current observation.

The learning loop has four key ideas:

1. `Q-network`
   The online network predicts the current Q-values and is updated by gradient descent.

2. `Replay buffer`
   Instead of learning from consecutive transitions directly, DQN stores transitions and samples random minibatches. This reduces temporal correlation and makes optimization more stable.

3. `Target network`
   A separate frozen copy of the Q-network is used to build the Bellman target. It is hard-updated every fixed number of environment steps.

4. `Epsilon-greedy exploration`
   The agent starts very exploratory and gradually shifts toward greedy action selection as epsilon decays.

For a transition `(s, a, r, s')`, vanilla DQN minimizes:

```text
target = r + gamma * max_a' Q_target(s', a')
loss = SmoothL1Loss(Q_online(s, a), target)
```

This implementation uses Gymnasium's split termination API correctly:

- `terminated=True`: the underlying MDP ended, so the Bellman target should not bootstrap
- `truncated=True`: the episode ended because of a time limit or wrapper condition, so the Bellman target still bootstraps from `next_obs`

In code, the target is:

```text
target = reward + gamma * (1 - terminated) * next_q
```

## Shared Environment Setup

There is already a virtual environment in `tabular_rl_cliffwalking/.venv`. To keep one venv across both projects, activate that same environment and install the extra DQN dependencies into it:

```bash
source tabular_rl_cliffwalking/.venv/bin/activate
uv pip install torch "gymnasium[classic-control,box2d]" matplotlib numpy
```

`LunarLander-v3` needs the Box2D extra, so make sure `gymnasium[box2d]` is installed.

## Train on CartPole

```bash
source tabular_rl_cliffwalking/.venv/bin/activate
python -m dqn_project.train --env-id CartPole-v1
```

CartPole defaults:

- `gamma=0.99`
- `lr=1e-3`
- `batch_size=64`
- `buffer_size=100000`
- `min_buffer_size=1000`
- `epsilon_start=1.0`
- `epsilon_end=0.05`
- `epsilon_decay_steps=20000`
- `target_update_freq=500`
- `max_episodes=500`
- network: `128, 128`

## Train on LunarLander

```bash
source tabular_rl_cliffwalking/.venv/bin/activate
python -m dqn_project.train --env-id LunarLander-v3
```

LunarLander defaults:

- `gamma=0.99`
- `lr=5e-4`
- `batch_size=128`
- `buffer_size=200000`
- `min_buffer_size=5000`
- `epsilon_start=1.0`
- `epsilon_end=0.05`
- `epsilon_decay_steps=100000`
- `target_update_freq=1000`
- `max_episodes=1500`
- network: `256, 256`

## Evaluate a Checkpoint

```bash
source tabular_rl_cliffwalking/.venv/bin/activate
python -m dqn_project.evaluate --checkpoint dqn_project/checkpoints/cartpole-v1/<run_name>/best.pt --episodes 10
```

## Logs and Saved Artifacts

Each run creates a folder under `dqn_project/checkpoints/<env_name>/<run_name>/` with:

- `config.json`
- `train_metrics.csv`
- `eval_metrics.csv` when evaluation runs
- `reward_curve.png`
- `loss_curve.png`
- `epsilon_curve.png`
- `eval_reward_curve.png` when evaluation runs
- `best.pt`
- `latest.pt`
- `summary.json`

## Useful CLI Overrides

- `--env-id CartPole-v1`
- `--env-id LunarLander-v3`
- `--device auto`
- `--hidden-dims 256 256`
- `--eval-every-episodes 10`
- `--eval-episodes 5`
- `--run-name debug_run`
- `--checkpoint-root /tmp/dqn_runs`

## Expected Learning Behavior

- CartPole should usually improve quickly and often reaches strong greedy scores within a few hundred episodes.
- LunarLander is much noisier and typically learns more slowly than CartPole.
- Early rewards can look chaotic even when the implementation is correct. Focus on evaluation curves and moving averages rather than any single episode.

## Common Debugging Issues

- Treating `terminated or truncated` as the Bellman mask. That incorrectly removes bootstrapping on time-limit truncations.
- Starting optimization before the replay buffer has enough data.
- Forgetting to update the target network.
- Evaluating with exploration still enabled instead of `greedy=True`.
- Feeding observations with inconsistent shapes or dtypes.
- Skipping gradient clipping and seeing unstable or exploding updates.
- Missing the Box2D dependency for `LunarLander-v3`.

## Extending to Double DQN

`agent.py` isolates bootstrap target computation in `_compute_next_q_values()`. That keeps the current vanilla DQN implementation simple and makes it straightforward to later switch to Double DQN by separating action selection from action evaluation.
