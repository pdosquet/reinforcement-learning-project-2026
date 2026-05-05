"""Plot drone trajectory vs waypoints for a trained agent.

Usage:
    python scripts/plot_trajectory.py --exp-id 22 --n-episodes 3
"""

import argparse
import sys
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import PyFlyt.gym_envs
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ALGO_CLASSES = {"ppo": PPO, "sac": SAC}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.wrappers import FlattenWaypointEnv, WaypointDistanceShaping  # noqa: E402

WRAPPER_CLASSES = {
    "FlattenWaypointEnv": FlattenWaypointEnv,
    "WaypointDistanceShaping": WaypointDistanceShaping,
}


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "PyFlyt-QuadX-Waypoints-v4" / "config.yml") as f:
        raw = yaml.unsafe_load(f)
    return dict(raw)


def make_env(env_kwargs, wrapper_specs):
    def _init():
        env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
        for spec in wrapper_specs:
            (name, kwargs), = spec.items()
            cls_name = name.rsplit(".", 1)[-1]
            env = WRAPPER_CLASSES[cls_name](env, **kwargs)
        return env
    return _init


def run_episode(model, vec_env, base_env):
    obs = vec_env.reset()
    # Snapshot waypoint targets before stepping (they shrink as drone reaches them)
    targets = np.array(base_env.unwrapped.waypoints.targets, copy=True)
    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, _ = vec_env.step(action)
        lin_pos = base_env.unwrapped.env.state(0)[-1]  # linear position (x, y, z)
        positions.append(lin_pos.copy())
        done = bool(done_arr[0])
    return np.array(positions), targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, required=True)
    parser.add_argument("--algo", default="ppo", choices=list(ALGO_CLASSES))
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--output", default="trajectory.png")
    args = parser.parse_args()

    exp_dir = PROJECT_ROOT / "results" / "zoo" / args.algo / f"PyFlyt-QuadX-Waypoints-v4_{args.exp_id}"
    config = load_config(exp_dir)

    env_kwargs = config.get("env_kwargs", {})
    wrapper_specs = config.get("env_wrapper", [])

    # Build vec env and load VecNormalize stats
    vec_env = DummyVecEnv([make_env(env_kwargs, wrapper_specs)])
    vec_norm_path = exp_dir / "PyFlyt-QuadX-Waypoints-v4" / "vecnormalize.pkl"
    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    base_env = vec_env.envs[0]  # unwrapped gym env (still wrapped w/ our wrappers)

    # Load model
    model_path = exp_dir / "best_model.zip"
    if not model_path.exists():
        model_path = exp_dir / "PyFlyt-QuadX-Waypoints-v4.zip"
    model = ALGO_CLASSES[args.algo].load(str(model_path), device="cpu")

    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_top = fig.add_subplot(122)

    for ep in range(args.n_episodes):
        positions, targets = run_episode(model, vec_env, base_env)
        color = plt.cm.tab10(ep)

        ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                  color=color, alpha=0.7, label=f"ep{ep+1} (len={len(positions)})")
        ax3d.scatter(*positions[0], color=color, marker="o", s=60)  # start
        ax3d.scatter(*positions[-1], color=color, marker="X", s=100)  # end
        for i, t in enumerate(targets):
            ax3d.scatter(*t, color=color, marker="*", s=150, edgecolors="black")

        ax_top.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.7)
        ax_top.scatter(positions[0, 0], positions[0, 1], color=color, marker="o", s=60)
        ax_top.scatter(positions[-1, 0], positions[-1, 1], color=color, marker="X", s=100)
        for t in targets:
            ax_top.scatter(t[0], t[1], color=color, marker="*", s=150, edgecolors="black")

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("Trajectories (3D) - o=start, X=end, *=waypoint")
    ax3d.legend()
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    ax_top.set_title("Top-down (x-y)")
    ax_top.axis("equal"); ax_top.grid(True)

    fig.tight_layout()
    fig.savefig(args.output, dpi=110)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
