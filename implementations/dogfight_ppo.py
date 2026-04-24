"""PPO-based dogfight agent trained with self-play.

Uses Stable-Baselines3 PPO. Self-play is implemented via a frozen-opponent
update schedule: every `opponent_update_freq` env steps the current policy is
copied and used as the new opponent.

Usage:
    python train_dogfight.py --config configs/dogfight/dogfight_ppo.yml --seed 0

The trained model is saved as:
    results/dogfight/dogfight_ppo_seed{seed}.zip          (SB3 checkpoint)
    results/dogfight/dogfight_ppo_seed{seed}_meta.json    (metadata)
"""

import json
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv

ALGORITHM_NAME = "dogfight_ppo"

# Fixed tournament environment kwargs (match the grader)
TOURNAMENT_ENV_KWARGS = dict(
    max_duration_seconds=60,
    agent_hz=30,
)

# Create multiple environments for self-play training
def make_multiple_env(n_envs: int = 4, opponent_policy=None, seed: int = 0):
    """ Create multiple environments for self-play training."""
    from scripts.dogfight_wrapper import DogfightSelfPlayEnv

    # Shared opponent container that all envs reference
    shared_opponent = {"policy": opponent_policy}

    def make_env_function():
        def init():
            env = DogfightSelfPlayEnv(
                team_size=1,
                opponent_policy=shared_opponent["policy"],
                flatten_observation=True,
                render_mode=None,
                **TOURNAMENT_ENV_KWARGS,
            )
            env.shared_opponent_ref = shared_opponent
            return env
        return init

    env_fns = [make_env_function() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env.shared_opponent = shared_opponent  # Attach for external updates
    return vec_env

# Create SB3 PPO model
def make_sb3_ppo(env, hyperparams: dict, seed: int, device: str = "cpu"):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=hyperparams["vf_coef"],
        max_grad_norm=hyperparams["max_grad_norm"],
        policy_kwargs={"net_arch": hyperparams["net_arch"]},
        verbose=1,
        seed=seed,
        device=device,
    )

# Train self-play PPO model
def train_selfplay(
    seed: int = 0,
    num_timesteps: int = 500_000,
    log_to_wandb: bool = False,
    output_dir: str | None = None,
    device: str = "cpu",
    hyperparams: dict | None = None,
) -> Path:
    hp = hyperparams or {}
    n_envs = int(hp.get("n_envs", 1))

    output_path = Path(output_dir or "results/dogfight")
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"dogfight_ppo_seed{seed}.zip"

    if log_to_wandb:
        try:
            import wandb
            wandb.init(
                project="pyflyt-rl",
                name=f"dogfight_ppo_seed{seed}",
                config={"algorithm": ALGORITHM_NAME, "seed": seed, "n_envs": n_envs, **hp},
                tags=["dogfight", "ppo", "selfplay"],
            )
        except Exception:
            log_to_wandb = False

    # Create multiple environments with shared opponent policy
    print(f"[dogfight] Creating {n_envs} parallel environments...")
    vec_env = make_multiple_env(n_envs=n_envs, opponent_policy=None, seed=seed)
    model = make_sb3_ppo(vec_env, hp, seed, device)

    warmup_steps = int(hp.get("warmup_steps", 0) or 0)
    opponent_update_freq = hp["opponent_update_freq"]
    steps_trained = 0
    opponent_snapshots = 0

    # When using n_envs, total steps collected per learn() call = n_steps * n_envs
    # So we need to adjust the update frequency to account for parallel collection
    effective_update_freq = opponent_update_freq // n_envs
    effective_warmup = warmup_steps // n_envs

    print(f"[dogfight] Starting self-play training  seed={seed}  steps={num_timesteps:,}  n_envs={n_envs}")
    if warmup_steps > 0:
        print(f"[dogfight] Warmup vs heuristic opponent for {warmup_steps:,} steps ({effective_warmup} steps per env)")
    print(f"[dogfight] Opponent updated every {opponent_update_freq:,} steps ({effective_update_freq} steps per env)")

    while steps_trained < effective_warmup:
        # Calculate exactly how many steps are left to reach the goal
        remaining_warmup = effective_warmup - steps_trained

        model.learn(
            total_timesteps=remaining_warmup,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        steps_trained += remaining_warmup * n_envs

    while steps_trained < num_timesteps:
        steps_this_round = min(effective_update_freq, num_timesteps // n_envs - steps_trained // n_envs)
        if steps_this_round <= 0:
            break

        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        steps_trained += steps_this_round * n_envs  # Scale by n_envs

        # Update opponent with a frozen copy of the current policy
        # Create a new policy instance and copy state dict (deepcopy fails on PyTorch tensors)
        def constant_lr_schedule():
            return hp["learning_rate"]

        frozen_policy = model.policy.__class__(
            observation_space=model.observation_space,
            action_space=model.action_space,
            lr_schedule=constant_lr_schedule,
            **model.policy_kwargs
        )
        frozen_policy.load_state_dict(model.policy.state_dict())
        frozen_policy.to(device)
        frozen_policy.eval()

        class _FrozenOpponent:
            def __init__(self, policy):
                self._policy = policy

            def predict(self, obs, deterministic=False):
                import torch
                obs_t = model.policy.obs_to_tensor(obs)[0]
                with torch.no_grad():
                    action = self._policy._predict(obs_t, deterministic=deterministic)
                return action.cpu().numpy(), {}

        # Update shared opponent reference so all vectorized envs see the new policy
        vec_env.shared_opponent["policy"] = _FrozenOpponent(frozen_policy)
        opponent_snapshots += 1
        print(
            f"[dogfight] steps={steps_trained:>10,}/{num_timesteps:,}  "
            f"opponent updated ({opponent_snapshots}x)"
        )

    model.save(str(checkpoint_path))
    vec_env.close()

    meta = {
        "algorithm": ALGORITHM_NAME,
        "seed": seed,
        "total_timesteps": steps_trained,
        "checkpoint_file": checkpoint_path.name,
        "hyperparams": hp,
        "tournament_env_kwargs": TOURNAMENT_ENV_KWARGS,
    }
    meta_path = output_path / f"dogfight_ppo_seed{seed}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return checkpoint_path
