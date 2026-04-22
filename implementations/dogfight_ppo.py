"""PPO-based dogfight agent trained with self-play via DogfightSelfPlayEnv.

Training pipeline
-----------------
Uses Stable-Baselines3 PPO.  Self-play is implemented via a frozen-opponent
update schedule: every `opponent_update_freq` env steps the current policy is
copied and used as the new opponent, gradually increasing difficulty.

Usage (via train_dogfight.py):
    python train_dogfight.py --timesteps 500000 --seed 0 --no-wandb

The trained model is saved as:
    results/dogfight/dogfight_ppo_seed{seed}.zip          (SB3 checkpoint)
    results/dogfight/dogfight_ppo_seed{seed}_meta.json    (metadata)

A ready-to-submit tournament file is written to:
    submissions/group_dogfight.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
ALGORITHM_NAME = "dogfight_ppo"

DOGFIGHT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [256, 256],
    # Vectorization - parallel envs for faster rollout collection
    "n_envs": 4,                      # number of parallel dogfight envs
    # Self-play
    "warmup_steps": 100_000,          # train vs stable heuristic opponent first
    "opponent_update_freq": 50_000,   # copy policy to opponent every N steps (per env, so total steps = n_envs * this)
}

# Fixed tournament environment kwargs (must match the grader exactly)
TOURNAMENT_ENV_KWARGS = dict(
    max_duration_seconds=60,
    agent_hz=30,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(opponent_policy=None, render_mode=None):
    """Create a DogfightSelfPlayEnv instance."""
    from scripts.dogfight_wrapper import DogfightSelfPlayEnv
    return DogfightSelfPlayEnv(
        team_size=1,
        opponent_policy=opponent_policy,
        flatten_observation=True,
        render_mode=render_mode,
        **TOURNAMENT_ENV_KWARGS,
    )


def _make_vec_env(n_envs: int = 4, opponent_policy=None, start_seed: int = 0):
    """Create a vectorized DummyVecEnv with multiple DogfightSelfPlayEnv instances.
    
    All envs share the same opponent policy reference for efficient self-play updates.
    """
    from scripts.dogfight_wrapper import DogfightSelfPlayEnv
    
    # Shared opponent container that all envs reference
    shared_opponent = {"policy": opponent_policy}
    
    def make_env_fn(env_idx):
        def _init():
            env = DogfightSelfPlayEnv(
                team_size=1,
                opponent_policy=shared_opponent["policy"],  # Initial value
                flatten_observation=True,
                render_mode=None,
                **TOURNAMENT_ENV_KWARGS,
            )
            # Store reference so we can update it later via shared_opponent dict
            env._shared_opponent_ref = shared_opponent
            return env
        return _init
    
    env_fns = [make_env_fn(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env.shared_opponent = shared_opponent  # Attach for external updates
    return vec_env


def _make_sb3_ppo(env, hp: dict, seed: int, device: str = "cpu"):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"],
        max_grad_norm=hp["max_grad_norm"],
        policy_kwargs={"net_arch": hp["net_arch"]},
        verbose=1,
        seed=seed,
        device=device,
    )


# ---------------------------------------------------------------------------
# Self-play training
# ---------------------------------------------------------------------------

def train_selfplay(
    seed: int = 0,
    num_timesteps: int = 500_000,
    log_to_wandb: bool = False,
    output_dir: str | None = None,
    device: str = "cpu",
    hyperparams_override: dict | None = None,
) -> Path:
    """Train a dogfight PPO agent with self-play.

    Returns the path to the saved SB3 .zip checkpoint.
    """
    hp = {**DOGFIGHT_HYPERPARAMS}
    if hyperparams_override:
        hp.update(hyperparams_override)
    
    n_envs = int(hp.get("n_envs", 1))

    output_path = Path(output_dir or (PROJECT_ROOT / "results" / "dogfight"))
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

    # Create vectorized envs with shared opponent policy reference
    print(f"[dogfight] Creating {n_envs} parallel environments...")
    vec_env = _make_vec_env(n_envs=n_envs, opponent_policy=None, start_seed=seed)
    model = _make_sb3_ppo(vec_env, hp, seed, device)

    warmup_steps = int(hp.get("warmup_steps", 0) or 0)
    opponent_update_freq = hp["opponent_update_freq"]
    steps_trained = 0
    opponent_snapshots = 0
    
    # When using n_envs, total steps collected per learn() call = n_steps * n_envs
    # So we need to adjust the update frequency to account for parallel collection
    effective_update_freq = opponent_update_freq // n_envs
    effective_warmup = warmup_steps // n_envs if warmup_steps > 0 else 0

    print(f"[dogfight] Starting self-play training  seed={seed}  steps={num_timesteps:,}  n_envs={n_envs}")
    if warmup_steps > 0:
        print(f"[dogfight] Warmup vs heuristic opponent for {warmup_steps:,} steps ({effective_warmup} steps per env)")
    print(f"[dogfight] Opponent updated every {opponent_update_freq:,} steps ({effective_update_freq} steps per env)")

    if effective_warmup > 0:
        steps_this_round = min(effective_warmup, num_timesteps - steps_trained)
        if steps_this_round > 0:
            model.learn(
                total_timesteps=steps_this_round,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            steps_trained += steps_this_round * n_envs  # Scale by n_envs

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
        def constant_lr_schedule(progress_remaining):
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

        if log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"steps_trained": steps_trained}, step=steps_trained)
            except Exception:
                pass

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

    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    print(f"[dogfight] Saved checkpoint → {checkpoint_path}")
    print(f"[dogfight] Saved metadata  → {meta_path}")
    return checkpoint_path


# ---------------------------------------------------------------------------
# load_model (tournament / evaluation interface)
# ---------------------------------------------------------------------------

def load_model(path: str | Path | None = None):
    """Load a trained dogfight PPO model.

    Args:
        path: path to a .zip SB3 checkpoint.  If None, looks for
              results/dogfight/dogfight_ppo_seed0.zip relative to the
              project root.

    Returns:
        An object with .predict(obs, deterministic=True) -> (action, info).
    """
    from stable_baselines3 import PPO

    if path is None:
        path = PROJECT_ROOT / "results" / "dogfight" / "dogfight_ppo_seed0.zip"

    return PPO.load(str(path))
