"""Shared helpers for model implementations used by train.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import gymnasium
import numpy as np
import PyFlyt.gym_envs  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_scripts_module(module_name: str):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import script module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_env_kwargs = _load_scripts_module("env_config").get_env_kwargs
FlattenWaypointEnv = _load_scripts_module("wrappers").FlattenWaypointEnv


TASK_EPISODE_LIMITS = {
    "hover": 500,
    "waypoints": 2000,
}


def create_env(task: str, flight_mode: int, render: bool = False):
    """Create the PyFlyt environment used by training and evaluation."""
    render_mode = "human" if render else None

    if task == "hover":
        return gymnasium.make(
            "PyFlyt/QuadX-Hover-v4",
            flight_mode=flight_mode,
            render_mode=render_mode,
        )

    if task == "waypoints":
        env_base = gymnasium.make(
            "PyFlyt/QuadX-Waypoints-v4",
            flight_mode=flight_mode,
            render_mode=render_mode,
            **get_env_kwargs("waypoints"),
        )
        return FlattenWaypointEnv(env_base, max_waypoints=4)

    raise ValueError(f"Unknown task: {task}")


def save_json(path: Path, payload: dict):
    """Write JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def init_wandb_run(enabled: bool, **kwargs):
    """Initialize a W&B run only when logging is enabled."""
    if not enabled:
        return None
    os.environ.setdefault("WANDB__DISABLE_STATS", "true")
    import wandb

    return wandb.init(**kwargs)


def log_wandb(enabled: bool, payload: dict, step=None):
    """Log metrics to W&B when enabled."""
    if not enabled:
        return
    import wandb

    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def finish_wandb(enabled: bool, summary=None):
    """Finish a W&B run when enabled."""
    if not enabled:
        return
    import wandb

    if summary is not None:
        if wandb.run is not None:
            wandb.run.summary.update(summary)
    wandb.finish()


def make_sb3_wandb_callback(enabled: bool):
    """Return a callback that logs SB3 training metrics to W&B every 500 steps.

    WandbCallback(gradient_save_freq=0) never logs by default — we need a
    custom callback that manually dumps SB3's internal logger to W&B.
    """
    if not enabled:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class _MetricLoggerCallback(BaseCallback):
        """Log SB3 training metrics + episode rewards to W&B every log_freq steps."""

        def __init__(self, log_freq: int = 500):
            super().__init__()
            self.log_freq = log_freq

        def _on_step(self) -> bool:
            if self.n_calls % self.log_freq != 0:
                return True
            import wandb
            if wandb.run is None:
                return True

            metrics = {
                k: v
                for k, v in self.model.logger.name_to_value.items()
                if v is not None
            }

            # SB3 only flushes ep_rew_mean every 4 episodes — read directly
            if len(self.model.ep_info_buffer) > 0:
                metrics["rollout/ep_rew_mean"] = np.mean(
                    [ep["r"] for ep in self.model.ep_info_buffer]
                )
                metrics["rollout/ep_len_mean"] = np.mean(
                    [ep["l"] for ep in self.model.ep_info_buffer]
                )

            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
            return True

    return _MetricLoggerCallback(log_freq=500)


def log_policy_video(enabled: bool, model, task: str, flight_mode: int, max_steps: int = 500):
    """Record one deterministic rollout and upload it as a W&B video.

    Creates a temporary rgb_array env, runs the policy greedily, and logs the
    resulting clip. Works headless: PyBullet uses its own software renderer for
    rgb_array mode and only needs a dummy DISPLAY value to initialise.
    Skips silently if anything goes wrong — never crashes training.
    """
    if not enabled:
        return

    # PyBullet needs DISPLAY set to initialise even in headless/rgb_array mode.
    os.environ.setdefault("DISPLAY", ":99")

    render_env = None
    try:
        if task == "hover":
            render_env = gymnasium.make(
                "PyFlyt/QuadX-Hover-v4",
                flight_mode=flight_mode,
                render_mode="rgb_array",
            )
        elif task == "waypoints":
            env_base = gymnasium.make(
                "PyFlyt/QuadX-Waypoints-v4",
                flight_mode=flight_mode,
                render_mode="rgb_array",
                **get_env_kwargs("waypoints"),
            )
            render_env = FlattenWaypointEnv(env_base, max_waypoints=4)
        else:
            return

        frames = []
        obs, _ = render_env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = render_env.step(action)
            frame = render_env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break

        if not frames:
            return

        import wandb

        # PyBullet returns RGBA (H, W, 4) — drop alpha, wandb.Video needs RGB.
        # wandb.Video expects (T, C, H, W) uint8.
        video = np.stack(frames)[..., :3].astype(np.uint8)  # (T, H, W, 3)
        video = np.transpose(video, (0, 3, 1, 2))            # (T, C, H, W)
        wandb.log({f"policy_video/{task}_mode{flight_mode}": wandb.Video(video, fps=24, format="mp4")})

    except Exception as exc:  # noqa: BLE001
        # Video logging is best-effort — never crash training over it
        print(f"  [video] skipped ({exc})")
    finally:
        if render_env is not None:
            render_env.close()


def load_artifact_metadata(artifact_dir) -> dict:
    """Load model_info.json from an artifact directory when available."""
    info_path = Path(artifact_dir) / "model_info.json"
    if not info_path.exists():
        return {}
    with open(info_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint_path(artifact_dir, metadata=None, default_name="checkpoint.pt", prefer_best=True) -> Path:
    """Return the checkpoint file inside an artifact directory."""
    path = Path(artifact_dir)
    metadata = metadata or {}
    best_checkpoint = path / "best_checkpoint.pt"
    if prefer_best and best_checkpoint.exists():
        return best_checkpoint
    return path / metadata.get("checkpoint_file", default_name)


def force_flight_mode(flight_mode: int):
    """Patch gymnasium.make so evaluate.py creates the environment with the saved mode."""
    original_make = gymnasium.make

    def patched_make(env_id, *args, **kwargs):
        kwargs["flight_mode"] = flight_mode
        return original_make(env_id, *args, **kwargs)

    gymnasium.make = patched_make


class ObservationCheckedModel:
    """Wrap a model and fail early when the observation dimension is wrong."""

    def __init__(self, model, expected_obs_dim=None, task=None):
        self.model = model
        self.expected_obs_dim = expected_obs_dim
        self.task = task

    def predict(self, obs, deterministic=True):
        if self.expected_obs_dim is not None:
            observed_dim = int(np.asarray(obs).shape[-1])
            if observed_dim != self.expected_obs_dim:
                task_hint = f" trained for '{self.task}'" if self.task else ""
                raise ValueError(
                    f"Observation size mismatch: got {observed_dim}, expected {self.expected_obs_dim}. "
                    f"This checkpoint was{task_hint}. "
                    "Use the generated results/models/*.py file that matches the environment."
                )
        return self.model.predict(obs, deterministic=deterministic)


def evaluate_policy(model, env, max_episode_length, n_episodes=5, deterministic=True):
    """Run a few episodes and return compact evaluation statistics."""
    rewards = []
    lengths = []

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=100 + episode_idx)
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_episode_length:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(float(total_reward))
        lengths.append(int(steps))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
    }