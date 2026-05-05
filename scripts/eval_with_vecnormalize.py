"""Evaluate a checkpoint using its saved VecNormalize stats.

Reproduces a clean post-hoc evaluation that does NOT suffer from rl_zoo3's
EvalCallback obs_rms drift issue.

Usage:
    python scripts/eval_with_vecnormalize.py --exp-id 32 --n-episodes 20
    python scripts/eval_with_vecnormalize.py --exp-id 32 --n-episodes 20 --stochastic
    python scripts/eval_with_vecnormalize.py --exp-id 32 --algo sac
"""

import argparse
import sys
from pathlib import Path

import gymnasium
import numpy as np
import PyFlyt.gym_envs  # noqa: F401
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.wrappers import (  # noqa: E402
    FlattenWaypointEnv,
    WaypointDistanceShaping,
    WaypointSumDistanceShaping,
)

ALGO_CLASSES = {"ppo": PPO, "sac": SAC}
WRAPPER_CLASSES = {
    "FlattenWaypointEnv": FlattenWaypointEnv,
    "WaypointDistanceShaping": WaypointDistanceShaping,
    "WaypointSumDistanceShaping": WaypointSumDistanceShaping,
}


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "PyFlyt-QuadX-Waypoints-v4" / "config.yml") as f:
        return dict(yaml.unsafe_load(f))


def make_env(env_kwargs, wrapper_specs):
    def _init():
        env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
        for spec in wrapper_specs:
            (name, kwargs), = spec.items()
            cls_name = name.rsplit(".", 1)[-1]
            env = WRAPPER_CLASSES[cls_name](env, **kwargs)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, required=True)
    parser.add_argument("--algo", default="ppo", choices=list(ALGO_CLASSES))
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (default: deterministic)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="best_model.zip",
                        help="Model file to load (default: best_model.zip)")
    args = parser.parse_args()

    exp_dir = PROJECT_ROOT / "results" / "zoo" / args.algo / f"PyFlyt-QuadX-Waypoints-v4_{args.exp_id}"
    if not exp_dir.exists():
        sys.exit(f"Run dir not found: {exp_dir}")

    config = load_config(exp_dir)
    env_kwargs = config.get("env_kwargs", {})
    wrapper_specs = config.get("env_wrapper", [])

    vec_env = DummyVecEnv([make_env(env_kwargs, wrapper_specs)])
    vec_env.seed(args.seed)

    vec_norm_path = exp_dir / "PyFlyt-QuadX-Waypoints-v4" / "vecnormalize.pkl"
    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from {vec_norm_path}")

    model_path = exp_dir / args.checkpoint
    if not model_path.exists():
        # rl_zoo3 also saves PyFlyt-QuadX-Waypoints-v4.zip at top level
        alt = exp_dir / "PyFlyt-QuadX-Waypoints-v4.zip"
        if alt.exists():
            model_path = alt
    print(f"Loading model from {model_path}")
    model = ALGO_CLASSES[args.algo].load(str(model_path), device="cpu")

    deterministic = not args.stochastic
    print(f"Mode: deterministic={deterministic}")
    print(f"env_kwargs: {dict(env_kwargs)}")
    print()

    rewards, lengths, waypoints_reached = [], [], []
    for ep in range(args.n_episodes):
        obs = vec_env.reset()
        total_reward, steps = 0.0, 0
        info = {}
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            total_reward += float(reward[0])
            steps += 1
            info = info_arr[0]
            if bool(done_arr[0]):
                break
        n_reached = info.get("num_targets_reached", 0)
        rewards.append(total_reward)
        lengths.append(steps)
        waypoints_reached.append(n_reached)
        print(f"  ep {ep+1:>2}: reward={total_reward:>8.2f}  steps={steps:>4}  waypoints_reached={n_reached}")

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    wp = np.array(waypoints_reached)
    print()
    print(f"Reward:    mean={rewards.mean():.2f}  std={rewards.std():.2f}  min={rewards.min():.2f}  max={rewards.max():.2f}")
    print(f"Length:    mean={lengths.mean():.1f}  min={lengths.min()}  max={lengths.max()}")
    print(f"Waypoints: mean={wp.mean():.2f}  (target: {env_kwargs.get('num_targets', '?')})")


if __name__ == "__main__":
    main()
