# Trained Models — Best Checkpoints (Table 6 of report)

Best checkpoints per (algorithm, environment) pair, as referenced in
Table 6 of the final report ("Final selected models per task and algorithm").

| Task      | Algo | Configuration                | Reward mean | Status      |
|-----------|------|------------------------------|-------------|-------------|
| Hover     | SAC  | `g099_utd3_mode6`            | ≈1100       | Converged   |
| Hover     | PPO  | `g099_ns4096_500k_mode6`     | ≈1100       | Converged   |
| Waypoints | PPO  | `s4_nws_sde_mode0` (3 seeds) | 800–900     | Best stable |
| Waypoints | SAC  | `g0995_utd3_500k_mode0`      | −100        | Failed      |

## Layout

```
hover_sac_g099_utd3_mode6/        best_model_seed{0,1}.zip
hover_ppo_g099_ns4096_500k_mode6/ best_model_seed{0,1}.zip
waypoints_ppo_s4_nws_sde_mode0/   best_model_seed{0,1,2}.zip + vecnormalize_seed{0,1,2}.pkl
waypoints_sac_g0995_utd3_500k_mode0/ best_model_seed{0,1,2}.zip
```

The `best_model_*.zip` files are SB3 checkpoints saved by RL Zoo 3 during
training (best evaluation episode reward over the run).

The PPO waypoints checkpoints are paired with `vecnormalize_*.pkl` (SB3
`VecNormalize` running statistics) — both must be loaded at inference time.

## Loading example (SB3)

```python
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Hover (no VecNormalize)
model = SAC.load("models/hover_sac_g099_utd3_mode6/best_model_seed0.zip")

# Waypoints PPO (with VecNormalize)
env = DummyVecEnv([make_env_fn])
env = VecNormalize.load(
    "models/waypoints_ppo_s4_nws_sde_mode0/vecnormalize_seed0.pkl", env
)
env.training = False
env.norm_reward = False
model = PPO.load("models/waypoints_ppo_s4_nws_sde_mode0/best_model_seed0.zip", env=env)
```
