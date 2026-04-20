# Tabular RL CliffWalking

This project implements the classic 4x12 CliffWalking task from scratch using tabular SARSA and Q-learning.

## Environment

- Grid size: 4 rows x 12 columns
- Start: bottom-left cell
- Goal: bottom-right cell
- Cliff: bottom-row cells between start and goal
- Actions:
  - `0 = up`
  - `1 = right`
  - `2 = down`
  - `3 = left`
- Rewards:
  - Normal move: `-1`
  - Cliff: `-100`, then the agent resets to the start state
  - Goal: `0`, and the episode ends
- State encoding:
  - `state = row * ncols + col`

## Update Equations

### SARSA

`Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))`

### Q-learning

`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

## Expected Behavior

In CliffWalking, Q-learning often converges to the shortest path that runs close to the cliff because it updates toward the greedy next-state value. That path is efficient, but it is risky during online interaction.

SARSA usually learns a safer path that stays farther away from the cliff because it updates using the action actually selected by the current epsilon-greedy policy. This tends to reflect the cost of risky exploration more directly.

## Project Files

- `env.py`: CliffWalking environment
- `agents.py`: tabular base agent, SARSA, and Q-learning
- `train.py`: training script and artifact saving
- `eval.py`: greedy evaluation and policy printing
- `utils.py`: plotting, metrics, and policy formatting helpers
- `requirements.txt`: exported dependency list

## Setup With uv

```bash
uv sync
```

## Train

```bash
uv run train.py --algo sarsa
uv run train.py --algo qlearning
```

Example with custom output directories:

```bash
uv run train.py --algo sarsa --output-dir outputs/sarsa_run
uv run train.py --algo qlearning --output-dir outputs/qlearning_run
```

## Evaluate

```bash
uv run eval.py --q-table outputs/sarsa/q_table.npy
uv run eval.py --q-table outputs/qlearning/q_table.npy
```

## Saved Artifacts

Each training run saves:

- `q_table.npy`
- `metrics.json`
- `reward_curve.png`
- `steps_curve.png`
- `falls_curve.png`
