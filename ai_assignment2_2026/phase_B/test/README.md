# Phase B — Experiments

## Overview

| File | Episodes | Repeats | Purpose |
|------|----------|---------|---------|
| `run_experiments.py` | 500 | 3 | Full results |
| `sanity_experiments.py` | 250 | 1 | Quick check |

Both scripts run on **CartPole-v1** and **MountainCar-v0**.

Algorithms:
- `mcts` — plain MCTS with UCT (no mentor)
- `inf_mcts` — informed MCTS with DQN mentor, noise levels `{0.0, 0.1, 0.3}`

---

## Run Full Experiments

```bash
# All algorithms
python run_experiments.py

# MCTS only
python run_experiments.py -a mcts

# Informed MCTS only (runs for each noise level)
python run_experiments.py -a inf_mcts

# With environment rendering (slow)
python run_experiments.py -a mcts --render
python run_experiments.py -a all -r
```

Results saved to `results/`.

---

## Run Sanity Experiments

```bash
# All algorithms
python sanity_experiments.py

# MCTS only
python sanity_experiments.py -a mcts

# Informed MCTS only
python sanity_experiments.py -a inf_mcts

# With environment rendering
python sanity_experiments.py -a mcts --render
python sanity_experiments.py -a all -r
```

Results saved to `results/sanity/`.

---

## Arguments

| Argument | Short | Values | Default | Description |
|----------|-------|--------|---------|-------------|
| `--algorithm` | `-a` | `all`, `mcts`, `inf_mcts` | `all` | Which algorithm to run |
| `--render` | `-r` | flag | off | Render environment visually |

---

## Output

For each domain (`CartPole-v1`, `MountainCar-v0`):

- `{domain}_combined.png` — all algorithms on same plot (only when running `all`)
- `{domain}__{label}.png` — individual plot per algorithm/noise level
- `{domain}__.npy` — raw reward arrays, shape `(NUM_REPEATS, NUM_EPISODES)`
