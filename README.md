## Training

Train any implementation entrypoint with `train.py`:

```bash
# Stable-Baselines3 PPO
python train.py --model implementations/ppo_lib.py --task waypoints --mode 6 --seed 0 --timesteps 500000 --no-wandb

# Scratch PPO implementation
python train.py --model implementations/ppo_implementation.py --task hover --mode 0 --seed 0 --timesteps 500000 --no-wandb

# Full benchmark sweep across tasks, modes and seeds
python train.py --model implementations/ppo_lib.py --benchmark --tasks hover waypoints --modes -1 0 4 6 7 --seeds 0 1 2 --timesteps 500000 --no-wandb

# Run two implementations sequentially with the same single-run config
python train.py --model implementations/ppo_lib.py implementations/ppo_implementation.py --task hover --mode 0 --seed 0 --timesteps 500000 --no-wandb
```

Single-run options:

- `--model` one or more implementation module paths (see **Model Interface** below)
- `--task` `hover` or `waypoints`
- `--mode` PyFlyt flight mode (e.g. `0` = attitude, `6` = velocity + yaw)
- `--seed` random seed
- `--timesteps` total environment steps
- `--no-wandb` disable Weights & Biases logging

Benchmark options (imply `--benchmark`):

- `--tasks` one or more tasks (default: `hover waypoints`)
- `--modes` one or more flight modes (default: `-1 0 4 6 7`)
- `--seeds` one or more seeds (default: `0 1 2`)

Checkpoints and metadata are saved to `results/artifacts/{algorithm}_{task}_mode{mode}_seed{seed}/`.

After each training run, `train.py` now generates:

- `results/models/{algorithm}_{task}_mode{mode}_seed{seed}.py` (submission/evaluation module)
- `results/artifacts/{algorithm}_{task}_mode{mode}_seed{seed}/` (`.pt` + `.json` files)

This means evaluation uses a `.py` model path directly and does not rely on a
`stable_baselines3` fallback path.

## Evaluation

Evaluate any generated submission module with `scripts/evaluate.py`:

```bash
# Evaluate a trained PPO run (recommended)
python scripts/evaluate.py --model results/models/ppo_hover_mode0_seed0.py --env hover

# Another trained run
python scripts/evaluate.py --model results/models/ppo_waypoints_mode6_seed0.py --env waypoints

# Evaluate the random baseline
python scripts/evaluate.py --model implementations/random_baseline.py --env hover --n_episodes 10

# Render live in PyBullet
python scripts/evaluate.py --model results/models/ppo_hover_mode0_seed0.py --env hover --render
```

The generated `.py` module internally points to its artifact directory and calls your implementation's `load_model(path)`.

## Implementations

The source folder is now `implementations/` to avoid confusion with generated
submission files under `results/models/`.

- `implementations/ppo_lib.py`: trainable PPO using Stable-Baselines3
- `implementations/ppo_implementation.py`: trainable scratch PPO entrypoint, including its PPO core
- `implementations/model_utils.py`: shared environment, artifact, and W&B helpers
- `implementations/random_baseline.py`: minimal random baseline and smallest train.py-compatible starting point

Every trainable module in `implementations/` must expose:

```python
ALGORITHM_NAME = "ppo"          # short name used in output paths

def train_model(task, flight_mode, seed, num_timesteps,
                log_to_wandb, output_dir): ...

def load_model(path=None):      # returns object with .predict(obs, deterministic) method
    ...
```

In the generated `results/models/*.py` files, `load_model(path=None)` receives the explicit
artifact directory path and should return an object exposing `predict(obs, deterministic=True)`.
