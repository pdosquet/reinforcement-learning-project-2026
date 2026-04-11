"""Minimal random baseline compatible with train.py and evaluate.py.

Every trainable implementation module must expose:
    - ALGORITHM_NAME
    - train_model(task, flight_mode, seed, num_timesteps, log_to_wandb, output_dir)
    - load_model(path=None)

This baseline does not learn. It exists as the smallest end-to-end example that
still satisfies the shared training and evaluation contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.model_utils import (
    TASK_EPISODE_LIMITS,
    ObservationCheckedModel,
    create_env,
    evaluate_policy,
    finish_wandb,
    force_flight_mode,
    init_wandb_run,
    load_artifact_metadata,
    log_wandb,
    save_json,
)


ALGORITHM_NAME = "random"
DEFAULT_CONFIG_FILE = "policy_config.json"
DEFAULT_ACTION_SPACE = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


class RandomPolicy:
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        del obs, deterministic
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        return self.rng.uniform(low=low, high=high).astype(np.float32), {}


def train_model(task, flight_mode, seed, num_timesteps, log_to_wandb, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    init_wandb_run(
        log_to_wandb,
        project="pyflyt-rl",
        name=f"{ALGORITHM_NAME}_{task}_mode{flight_mode}_seed{seed}",
        config={
            "algorithm": ALGORITHM_NAME,
            "task": task,
            "flight_mode": flight_mode,
            "seed": seed,
            "num_timesteps": num_timesteps,
        },
        tags=[ALGORITHM_NAME, task, f"mode_{flight_mode}"],
        sync_tensorboard=False,
    )

    env = create_env(task, flight_mode)
    try:
        policy = RandomPolicy(env.action_space, seed=seed)
        stats = evaluate_policy(policy, env, TASK_EPISODE_LIMITS[task], n_episodes=3)
        save_json(
            output_path / DEFAULT_CONFIG_FILE,
            {
                "algorithm": ALGORITHM_NAME,
                "task": task,
                "flight_mode": flight_mode,
                "seed": seed,
                "observation_dim": int(env.observation_space.shape[0]),
            },
        )
        summary = {
            "checkpoint_file": DEFAULT_CONFIG_FILE,
            "eval_reward_mean": stats["mean_reward"],
            "eval_reward_std": stats["std_reward"],
        }
        log_wandb(log_to_wandb, summary)
        finish_wandb(log_to_wandb)
        return summary
    finally:
        env.close()


def load_model(path=None):
    metadata = {}
    config = {"seed": 42, "observation_dim": None}

    if path is not None:
        resolved_path = Path(path)
        if resolved_path.is_dir():
            metadata = load_artifact_metadata(resolved_path)
            config_path = resolved_path / DEFAULT_CONFIG_FILE
            if "flight_mode" in metadata:
                force_flight_mode(metadata["flight_mode"])
        else:
            config_path = resolved_path

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                config.update(json.load(handle))

    policy = RandomPolicy(DEFAULT_ACTION_SPACE, seed=config.get("seed", 42))
    return ObservationCheckedModel(
        policy,
        expected_obs_dim=config.get("observation_dim"),
        task=metadata.get("task") or config.get("task"),
    )