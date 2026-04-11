"""Stable-Baselines3 PPO model implementation usable by train.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before SB3/TF are imported.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.model_utils import (
    ObservationCheckedModel,
    create_env,
    finish_wandb,
    force_flight_mode,
    init_wandb_run,
    load_artifact_metadata,
    log_wandb,
)


ALGORITHM_NAME = "ppo_lib"
FINAL_MODEL_FILE = "final_model.zip"
BEST_MODEL_FILE = "best_model.zip"

def train_model(task, flight_mode, seed, num_timesteps, log_to_wandb, output_dir):
    """Train PPO with stable-baselines3 and save artifacts in output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    init_wandb_run(
        log_to_wandb,
        project="pyflyt-rl",
        name=f"{ALGORITHM_NAME}_{task}_mode{flight_mode}_seed{seed}",
        config={
            "algorithm": ALGORITHM_NAME,
            "task": task,
            "library": "stable-baselines3",
        },
        tags=[ALGORITHM_NAME, task, f"mode_{flight_mode}"],
        sync_tensorboard=False,
    )

    env = create_env(task, flight_mode)
    eval_env = Monitor(create_env(task, flight_mode))

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=0,
            device="cpu",
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_path),
            log_path=str(output_path / "eval_logs"),
            eval_freq=max(1000, num_timesteps // 10),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=0,
        )
        model.learn(total_timesteps=num_timesteps, callback=eval_callback, progress_bar=False)
        model.save(str(output_path / "final_model"))

        checkpoint_file = BEST_MODEL_FILE if (output_path / BEST_MODEL_FILE).exists() else FINAL_MODEL_FILE
        best_reward = None
        if eval_callback.best_mean_reward != float("-inf"):
            best_reward = float(eval_callback.best_mean_reward)

        summary = {
            "checkpoint_file": checkpoint_file,
            "eval_reward_mean": best_reward,
        }
        log_wandb(log_to_wandb, summary)
        finish_wandb(log_to_wandb)
        return summary
    finally:
        env.close()
        eval_env.close()


def load_model(path=None):
    """Load a stable-baselines3 PPO policy from a file or artifact directory."""
    if path is None:
        raise ValueError("A saved model file or artifact directory path is required")

    metadata = {}
    resolved_path = Path(path)

    if resolved_path.is_dir():
        metadata = load_artifact_metadata(resolved_path)
        if "flight_mode" in metadata:
            force_flight_mode(metadata["flight_mode"])

        if metadata.get("checkpoint_file"):
            checkpoint_path = resolved_path / metadata["checkpoint_file"]
        elif (resolved_path / BEST_MODEL_FILE).exists():
            checkpoint_path = resolved_path / BEST_MODEL_FILE
        else:
            checkpoint_path = resolved_path / FINAL_MODEL_FILE
    else:
        checkpoint_path = resolved_path

    model = PPO.load(str(checkpoint_path), device="cpu")
    expected_obs_dim = int(model.observation_space.shape[0])
    return ObservationCheckedModel(model, expected_obs_dim=expected_obs_dim, task=metadata.get("task"))
