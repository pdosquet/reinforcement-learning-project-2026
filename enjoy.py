"""Replay a trained policy using rl_zoo3.enjoy.

Loads the best checkpoint from results/zoo for the given config and flight mode,
then renders one episode in the PyFlyt simulator.

Usage:
    # Latest run for a config + mode
    python enjoy.py --config configs/sac_hover_g098_utd3.yml --mode 0

    # Pick a specific experiment ID (shown in results/zoo/<algo>/<env>_<id>/)
    python enjoy.py --config configs/sac_hover_g098_utd3.yml --mode 0 --exp-id 2

    # Run for more steps, stochastic policy
    python enjoy.py --config configs/sac_hover_g098_utd3.yml --mode 0 --steps 2000 --stochastic
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

TASK_TO_ENV_ID = {
    "hover": "PyFlyt/QuadX-Hover-v4",
    "waypoints": "PyFlyt/QuadX-Waypoints-v4",
}


def load_meta(config_path: Path) -> tuple[str, str, str]:
    """Return (algo, task, env_id) from a config YAML."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    meta = raw.get("_meta", {})
    algo = meta.get("algo")
    task = meta.get("task")
    if not algo or not task:
        raise ValueError(f"{config_path}: missing _meta.algo or _meta.task")
    if task not in TASK_TO_ENV_ID:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_TO_ENV_ID)}")
    return algo, task, TASK_TO_ENV_ID[task]


def main():
    parser = argparse.ArgumentParser(
        description="Replay a trained policy with rl_zoo3.enjoy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True,
                        help="Config YAML used to train (e.g. configs/sac_hover_g098_utd3.yml)")
    parser.add_argument("--mode", type=int, required=True,
                        help="Flight mode the policy was trained with")
    parser.add_argument("--exp-id", type=int, default=0,
                        help="Experiment ID in results/zoo (0 = latest, default: 0)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Max steps to render (default: 1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (default: deterministic)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    algo, task, env_id = load_meta(config_path)

    print(f"Config:  {config_path}")
    print(f"Algo:    {algo}  |  Env: {env_id}  |  Mode: {args.mode}")
    print(f"Exp ID:  {args.exp_id or 'latest'}")

    cmd = [
        sys.executable, "-m", "rl_zoo3.enjoy",
        "--algo", algo,
        "--env", env_id,
        "-f", str(PROJECT_ROOT / "results" / "zoo"),
        "--gym-packages", "PyFlyt.gym_envs",
        "--env-kwargs", f"flight_mode:{args.mode}",
        "--exp-id", str(args.exp_id),
        "--seed", str(args.seed),
        "-n", str(args.steps),
        "--load-best",
        "--device", args.device,
    ]
    if not args.stochastic:
        cmd.append("--deterministic")

    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    print(f"\nRunning: {' '.join(cmd)}\n")
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
