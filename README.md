# INFO8003 — Reinforcement Learning Project

UAV control via deep reinforcement learning using PyFlyt.

## Provided Scripts

| Script                           | Purpose                                                    |
| -------------------------------- | ---------------------------------------------------------- |
| `scripts/env_config.py`          | Environment parameters (waypoint overrides)                |
| `scripts/wrappers.py`            | `FlattenWaypointEnv` — flattens dict observations          |
| `scripts/dogfight_wrapper.py`    | `DogfightSelfPlayEnv` — multi-agent → single-agent wrapper |
| `scripts/evaluate.py`            | Evaluate a trained model on Hover or Waypoints             |
| `scripts/tournament.py`          | Elo-rated dogfight tournament                              |
| `scripts/submission_template.py` | Tournament submission template                             |

## Setup

```bash
pip install -r requirements.txt
wandb login  # Optional, only needed when W&B logging is enabled
```

## Training

Train any implementation entrypoint with `train.py`:

```bash
python train.py --model implementations/random_baseline.py --task waypoints --mode 6 --seed 0 --timesteps 500000 --no-wandb

python train.py --model implementations/random_baseline.py --benchmark --tasks waypoints --modes 7 --seeds 0 --timesteps 50000 --no-wandb
```

Single-run options:

- `--model` one or more implementation module paths (see **Model Interface** below)
- `--task` `hover` or `waypoints`
- `--mode` PyFlyt flight mode
- `--seed` random seed
- `--timesteps` total environment steps
- `--no-wandb` disable Weights & Biases logging

Benchmark options (imply `--benchmark`):

- `--tasks` one or more tasks (default: `hover waypoints`)
- `--modes` one or more flight modes (default: `-1 0 4 6 7`)
- `--seeds` one or more seeds (default: `0 1 2`)

Checkpoints and metadata are saved to `results/artifacts/{algorithm}_{task}_mode{mode}_seed{seed}/`.

After each training run, `train.py` generates:

- `results/models/{algorithm}_{task}_mode{mode}_seed{seed}.py` (submission/evaluation module)
- `results/artifacts/{algorithm}_{task}_mode{mode}_seed{seed}/` (`.pt` + `.json` files)

## Evaluation

Evaluate any generated submission module with `scripts/evaluate.py`:

```bash
python scripts/evaluate.py --model results/models/random_baseline_hover_mode0_seed0.py --env hover

python scripts/evaluate.py --model results/models/random_baseline_hover_mode0_seed0.py --env hover --n_episodes 10 --render
```

The generated `.py` module internally points to its artifact directory and calls your implementation's `load_model(path)`.

## Implementations

- `implementations/model_utils.py`: shared environment, artifact, and W&B helpers
- `implementations/random_baseline.py`: minimal random baseline and smallest train.py-compatible starting point

Every trainable module in `implementations/` must expose:

```python
ALGORITHM_NAME = "random"

def train_model(task, flight_mode, seed, num_timesteps,
                log_to_wandb, output_dir): ...

def load_model(path=None):
    ...
```

In the generated `results/models/*.py` files, `load_model(path=None)` receives the explicit
artifact directory path and should return an object exposing `predict(obs, deterministic=True)`.
