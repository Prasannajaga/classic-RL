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

For both methods, the tabular update has the same general form:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \, \delta_t
$$

where the temporal-difference error is:

$$
\delta_t = \text{target}_t - Q(s_t, a_t)
$$

Here:

- $s_t$: current state
- $a_t$: action taken in the current state
- $r_{t+1}$: reward received after taking $a_t$
- $s_{t+1}$: next state
- $\alpha$: learning rate
- $\gamma$: discount factor

### SARSA

SARSA is an on-policy method because it updates using the action actually chosen next by the current policy.

$$
\text{target}_t^{\text{SARSA}} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})
$$

$$
\delta_t^{\text{SARSA}} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
$$

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right)
$$

### Q-learning

Q-learning is an off-policy method because it updates toward the best possible next action value, regardless of the action actually taken during exploration.

$$
\text{target}_t^{\text{Q-learning}} = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')
$$

$$
\delta_t^{\text{Q-learning}} = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)
$$

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
$$

For terminal states, there is no future value term, so the target becomes:

$$
\text{target}_t = r_{t+1}
$$

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

## Train & Evaluate

```bash
uv run train.py --algo sarsa
uv run train.py --algo qlearning
``` 

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
