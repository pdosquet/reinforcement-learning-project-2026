# Info8003 — Reinforcement Learning Project 2026

Quadrotor control experiments on [PyFlyt](https://github.com/jjshoots/PyFlyt) environments
(`QuadX-Hover-v4` and `QuadX-Waypoints-v4`) using PPO and SAC, plus a self-play dogfight
agent for `MAFixedwingDogfightEnvV2`.

---

## 1. Project Overview

The project trains and evaluates quadrotor flight policies using:

- **PPO (scratch)** — a hand-written PPO agent in `implementations/ppo_implementation.py`,
  configurable via YAML and trainable through `train_zoo.py`
- **PPO (SB3 via Zoo)** — Stable-Baselines3 PPO driven by `rl_zoo3.train`
- **SAC (SB3 via Zoo)** — Stable-Baselines3 SAC driven by `rl_zoo3.train`
- **Dogfight PPO** — SB3 PPO trained in self-play via `DogfightSelfPlayEnv`

## 2. Repository Structure

```
implementations/
    model_utils.py          # Shared env creation, W&B helpers, artifact I/O
    ppo_implementation.py   # Scratch PPO (ActorCritic + GAE + clipped surrogate)
    dogfight_ppo.py         # Self-play PPO for the dogfight tournament

configs/
    ppo/                    # SB3 PPO configs via rl_zoo3
        mode0/              # hover mode 0: gamma x n_steps grid (100k / 500k)
        mode6/              # hover mode 6: same grid
    ppo_scratch/            # Scratch PPO configs (impl: ppo_implementation.py)
        ppo_scratch_hover_mode0.yml
        ppo_scratch_hover_mode6.yml
        ppo_scratch_waypoints_mode6.yml
    sac/                    # SB3 SAC configs via rl_zoo3
        mode0/              # hover + waypoints mode 0
        mode6/              # hover + waypoints mode 6

scripts/
    env_config.py           # Centralised env kwargs (waypoints dome size, etc.)
    wrappers.py             # FlattenWaypointEnv, ActionRepeat, reward-shaping wrappers
    evaluate.py             # Standalone evaluation script
    tournament.py           # Elo-rated round-robin dogfight tournament
    dogfight_wrapper.py     # DogfightSelfPlayEnv (Gymnasium wrapper over PettingZoo)
    submission_template.py  # Tournament submission template

results/
    artifacts/              # Scratch PPO checkpoints (.pt) + metadata (.json)
    zoo/                    # RL Zoo 3 outputs (tensorboard, best_model.zip, etc.)
    dogfight/               # Dogfight checkpoints (.zip) + metadata (.json)

submissions/
    (manual tournament submissions)

train_zoo.py                # Unified training entry-point (Zoo + scratch PPO)
train_dogfight.py           # Self-play dogfight training script
enjoy.py                    # RL Zoo 3 replay / rendering
RESULTS_ANALYSIS.md         # Per-experiment notes, tables, and conclusions
```

---

## 3. Installation

```bash
pip install -r requirements.txt
```

## 4. Environments

### Flight modes (hover & waypoints)

| Mode | Action semantics                           | Difficulty |
| ---- | ------------------------------------------ | ---------- |
| `0`  | Angular velocity + thrust                  | Hard       |
| `6`  | Ground velocity + yaw rate + vertical vel. | Easy       |

### Dogfight environment

The grader uses a fixed configuration:

```python
MAFixedwingDogfightEnvV2(
    team_size=1, assisted_flight=True,
    flatten_observation=True, max_duration_seconds=60, agent_hz=30,
)
```

assisted_flight=True, # Simplified control: velocity commands instead of raw motor control

---

## 5. Training — Hover & Waypoints

A single entry-point handles both SB3/Zoo algorithms and the scratch PPO:

```bash
python train_zoo.py --config <path/to/config.yml> --mode <int> --seed <int>
```

The `train_zoo.py` script supports several training configurations:

| Option             | Description                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| **SB3 Algorithms** | Use `--config` pointing to `configs/sac/` or `configs/ppo/` for Stable-Baselines3 PPO/SAC via rl_zoo3 |
| **Scratch PPO**    | Use `--config` pointing to `configs/ppo_scratch/` — routes to `implementations/ppo_implementation.py` |

### Examples

```bash
# SAC hover, mode 0, seed 0
python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0

# SAC hover, without W&B
python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0 --no-wandb

# Override timesteps
python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0 --timesteps 200000

# SB3 PPO hover, mode 0, both seeds
python train_zoo.py --config configs/ppo/mode0/ppo_hover_g098_ns2048_500k_mode0.yml --mode 0 --seeds 0 1

# Scratch PPO hover, mode 0 (via _meta.impl routing)
python train_zoo.py --config configs/ppo_scratch/ppo_scratch_hover_mode0.yml --mode 0 --seed 0

# Scratch PPO hover, mode 6
python train_zoo.py --config configs/ppo_scratch/ppo_scratch_hover_mode6.yml --mode 6 --seed 0

# Scratch PPO waypoints, mode 6
python train_zoo.py --config configs/ppo_scratch/ppo_scratch_waypoints_mode6.yml --mode 6 --seed 0

# Sweep multiple modes and seeds
python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --modes 0 6 --seeds 0 1 2
```

### Flags

| Flag          | Description                             |
| ------------- | --------------------------------------- |
| `--config`    | Path to a YAML config                   |
| `--mode`      | Single flight mode (used with `--seed`) |
| `--seed`      | Single seed (default: `0`)              |
| `--modes`     | Multiple flight modes for a sweep       |
| `--seeds`     | Multiple seeds for a sweep              |
| `--timesteps` | Override `n_timesteps` from the YAML    |
| `--no-wandb`  | Disable W&B tracking                    |
| `--device`    | `cpu` or `cuda` (default: `cuda`)       |

### Outputs

| Backend       | Output location                                    |
| ------------- | -------------------------------------------------- |
| rl_zoo3 (SB3) | `results/zoo/<algo>/<env_id>_<exp_id>/`            |
| Scratch PPO   | `results/artifacts/<config_stem>_mode<M>_seed<S>/` |

---

## 6. Training — Dogfight (Self-Play)

`train_dogfight.py` trains a PPO agent in `DogfightSelfPlayEnv` with a frozen-opponent
self-play schedule. Uses YAML configs (like `train_zoo.py`) for hyperparameters.

```bash
# Basic run with default config
python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0

# No W&B logging
python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0 --no-wandb

# Multiple seeds
python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seeds 0 1 2

# Override timesteps from command line
python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0 --timesteps 1000000
```

| Flag           | Description                         | Default             |
| -------------- | ----------------------------------- | ------------------- |
| `--config`     | YAML config path (required)         | —                   |
| `--timesteps`  | Override training steps from config | from config         |
| `--seed`       | Random seed                         | 0                   |
| `--seeds`      | Multiple seeds (overrides `--seed`) | —                   |
| `--device`     | `cpu` or `cuda`                     | `cpu`               |
| `--no-wandb`   | Disable W&B                         | off                 |
| `--output-dir` | Output directory for checkpoints    | `results/dogfight/` |

### Outputs

| File                                              | Purpose                       |
| ------------------------------------------------- | ----------------------------- |
| `results/dogfight/dogfight_ppo_seed{S}.zip`       | SB3 PPO checkpoint            |
| `results/dogfight/dogfight_ppo_seed{S}_meta.json` | Metadata (hyperparams, steps) |

---

## 7. Evaluation

```bash
# SAC hover (Zoo checkpoint)
python scripts/evaluate.py \
    --model results/zoo/sac/PyFlyt-QuadX-Hover-v4_1/best_model.zip \
    --env hover

# Scratch PPO hover
python scripts/evaluate.py \
    --model results/artifacts/ppo_scratch_hover_mode0_mode0_seed0/best_checkpoint.pt \
    --env hover --mode 0

# Waypoints with rendering
python scripts/evaluate.py \
    --model results/zoo/ppo/PyFlyt-QuadX-Waypoints-v4_1/best_model.zip \
    --env waypoints --render
```

| Flag                               | Description                                    |
| ---------------------------------- | ---------------------------------------------- |
| `--model`                          | `.py` submission, SB3 `.zip`, or scratch `.pt` |
| `--env`                            | `hover` or `waypoints`                         |
| `--mode`                           | Flight mode (default: `0`)                     |
| `--n_episodes`                     | Number of evaluation episodes (default: `20`)  |
| `--render`                         | Open a PyBullet window                         |
| `--deterministic` / `--stochastic` | Action selection mode                          |

---

## 8. Replay / Enjoy

```bash
# Replay latest SAC hover run
python enjoy.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0

# Pick experiment ID 2, stochastic, 2000 steps
python enjoy.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --exp-id 2 --steps 2000 --stochastic
```

---

## 9. Implementations

### `implementations/ppo_implementation.py` — Scratch PPO

A fully hand-written PPO agent:

- `ActorNetwork`: 2-layer MLP, Tanh activations, Gaussian head with learned log-std
- `CriticNetwork`: 2-layer MLP value function
- `PPOAgent`: GAE advantages, clipped surrogate, entropy bonus, gradient clipping
- `PPOTrainer`: on-policy trajectory collection, multi-epoch updates
- Accepts optional `hyperparams_override` dict from `train_zoo.py` YAML configs

Default hyperparams:

| Param           | Value  |     | Param               | Value  |
| --------------- | ------ | --- | ------------------- | ------ |
| `learning_rate` | `3e-4` |     | `hidden_size`       | `64`   |
| `gamma`         | `0.99` |     | `trajectory_length` | `2048` |
| `gae_lambda`    | `0.95` |     | `epochs_per_update` | `10`   |
| `clip_ratio`    | `0.2`  |     | `entropy_coef`      | `0.01` |

### `implementations/dogfight_ppo.py` — Dogfight PPO

SB3 PPO trained with `DogfightSelfPlayEnv`:

- Every `opponent_update_freq` steps the current policy is frozen as the new opponent
- Default network: 2 x 256 MLP (`net_arch: [256, 256]`)
- Saves as `.zip` for direct tournament loading
- `load_model(path)` returns the SB3 PPO model

### `implementations/model_utils.py` — Shared Utilities

| Symbol                                      | Purpose                                 |
| ------------------------------------------- | --------------------------------------- |
| `create_env(task, mode, render)`            | Instantiates the correct env + wrappers |
| `evaluate_policy(model, env, max_steps, n)` | Returns reward statistics               |
| `ObservationCheckedModel`                   | Validates obs shape at inference        |
| `init_wandb_run / log_wandb / finish_wandb` | W&B lifecycle                           |
| `make_sb3_wandb_callback`                   | SB3 callback logging every 500 steps    |
| `TASK_EPISODE_LIMITS`                       | `{"hover": 500, "waypoints": 2000}`     |

---

## 10. Configuration Files

### Schema for Zoo / SB3 configs

```yaml
_meta:
   algo: sac # passed to rl_zoo3 --algo
   task: hover # determines env_id

PyFlyt/QuadX-Hover-v4:
   n_timesteps: 1.0e5
   policy: MlpPolicy
   learning_rate: 3.0e-4
   # ... standard rl_zoo3 hyperparameters
```

### Schema for scratch PPO configs

```yaml
_meta:
   algo: ppo_scratch
   task: hover
   impl: implementations/ppo_implementation.py # routing key

PyFlyt/QuadX-Hover-v4:
   n_timesteps: 5.0e5
   learning_rate: 3.0e-4
   n_steps: 2048 # -> trajectory_length
   gamma: 0.98
   gae_lambda: 0.95
   clip_range: 0.2 # -> clip_ratio
   ent_coef: 0.01 # -> entropy_coef
   vf_coef: 0.5 # -> value_loss_coef
   max_grad_norm: 0.5
   hidden_size: 64
   n_epochs: 10 # -> epochs_per_update
```

When `_meta.impl` is set, `train_zoo.py` calls `train_model(..., hyperparams_override={...})`
directly instead of launching rl_zoo3.

### Available configs

| Path                   | Algo        | Task              | Variants                 |
| ---------------------- | ----------- | ----------------- | ------------------------ |
| `configs/ppo/mode0/`   | PPO SB3     | hover             | gamma x n_steps x budget |
| `configs/ppo/mode6/`   | PPO SB3     | hover             | gamma x n_steps x budget |
| `configs/ppo_scratch/` | PPO scratch | hover, waypoints  | mode 0/6                 |
| `configs/sac/mode0/`   | SAC         | hover + waypoints | gamma x UTD              |
| `configs/sac/mode6/`   | SAC         | hover + waypoints | gamma x UTD              |
| `configs/dogfight/`    | PPO         | dogfight          | self-play schedule       |

## 11. Tournament Submission

### Dogfight Checkpoint

Trained models are saved to `results/dogfight/` as SB3 checkpoints.

```bash
# Train a model
python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0 --timesteps 1000000 --no-wandb

# Verify checkpoint
python -c "from stable_baselines3 import PPO; print(PPO.load('results/dogfight/dogfight_ppo_seed0.zip'))"

# Step 3: run local tournament
python scripts/tournament.py submissions/
```

Tournament environment (fixed, matches grader):

```python
MAFixedwingDogfightEnvV2(
    team_size=1, assisted_flight=True,
    flatten_observation=True, max_duration_seconds=60, agent_hz=30,
)
```

### Visualization

Live rendering of trained agents for qualitative analysis:

```bash
# Watch your agent vs heuristic opponent
python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent heuristic

# Watch vs another trained model
python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent-model submissions/baseline_heuristic.py

# Watch vs random opponent
python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent random

# Run multiple matches back-to-back
python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent heuristic --matches 5

# Use specific seed
python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent heuristic --seed 42
```
