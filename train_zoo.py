"""RL Zoo 3 training wrapper for PyFlyt experiments.

Reads a YAML config from configs/, injects the flight mode into env_kwargs,
and delegates to rl_zoo3.train via subprocess.

If _meta.impl is set in the YAML, the config's hyperparameters are forwarded
to that Python implementation's train_model() directly instead of rl_zoo3.

Single run (Zoo/SB3 algos):
    python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0
    python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0 --no-wandb
    python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --mode 0 --seed 0 --timesteps 200000

Single run (scratch PPO via _meta.impl):
    python train_zoo.py --config configs/ppo_scratch/ppo_scratch_hover_mode0.yml --mode 0 --seed 0

Sweep over multiple modes / seeds:
    python train_zoo.py --config configs/sac/mode0/sac_hover_g098_utd3_mode0.yml --modes 0 6 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# Keys that appear in SB3/Zoo YAML configs and map to scratch PPO hyperparams.
_SB3_TO_SCRATCH_KEY_MAP = {
    "learning_rate": "learning_rate",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip_range": "clip_ratio",
    "ent_coef": "entropy_coef",
    "vf_coef": "value_loss_coef",
    "max_grad_norm": "max_grad_norm",
    "n_steps": "trajectory_length",
    "n_epochs": "epochs_per_update",
    "hidden_size": "hidden_size",
    "epochs_per_update": "epochs_per_update",
}

TASK_TO_ENV_ID = {
    "hover": "PyFlyt/QuadX-Hover-v4",
    "waypoints": "PyFlyt/QuadX-Waypoints-v4",
}


def load_config(config_path: Path) -> tuple[str, str, str, dict, str | None]:
    """Load a config YAML and return (algo, task, env_id, hyperparams_dict, impl_path).

    The YAML has a `_meta` block with `algo`, `task`, and optionally `impl` keys,
    plus one key per env ID containing the hyperparameters.  The `_meta` block is
    stripped before the hyperparams dict is returned.

    When `_meta.impl` is set, `impl_path` is the path to a custom Python
    implementation module; rl_zoo3 is bypassed and train_model() is called directly.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    meta = raw.pop("_meta", {})
    algo = meta.get("algo")
    task = meta.get("task")
    impl_path = meta.get("impl")  # optional: path to custom Python module

    if not algo or not task:
        raise ValueError(
            f"{config_path}: missing _meta.algo or _meta.task. "
            "Each config must declare its algorithm and task."
        )
    if task not in TASK_TO_ENV_ID:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_TO_ENV_ID)}")

    env_id = TASK_TO_ENV_ID[task]
    if env_id not in raw:
        raise ValueError(
            f"{config_path}: expected a '{env_id}' key for task '{task}', but it was not found."
        )

    return algo, task, env_id, raw, impl_path


def _load_impl_module(impl_path: str):
    """Dynamically import a Python implementation module from a file path."""
    resolved = PROJECT_ROOT / impl_path
    if not resolved.exists():
        resolved = Path(impl_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Implementation module not found: {impl_path}")
    spec = importlib.util.spec_from_file_location("_impl_module", resolved)
    module = importlib.util.module_from_spec(spec)
    # Make the project root importable inside the module
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(module)
    return module


def run_single_impl(
    impl_path: str,
    task: str,
    env_id: str,
    hyperparams: dict,
    config_stem: str,
    mode: int,
    seed: int,
    timesteps: int | None,
    no_wandb: bool,
) -> int:
    """Route training to a custom Python implementation instead of rl_zoo3.

    Reads hyperparameters from the YAML, maps SB3-style keys to the scratch
    PPO convention, and calls module.train_model().
    Returns 0 on success, 1 on failure.
    """
    module = _load_impl_module(impl_path)

    env_block: dict = hyperparams.get(env_id, {})
    num_timesteps = timesteps if timesteps is not None else int(env_block.get("n_timesteps", 500_000))

    # Map SB3 YAML keys → scratch PPO hyperparams keys
    hyperparams_override: dict = {}
    for yaml_key, scratch_key in _SB3_TO_SCRATCH_KEY_MAP.items():
        if yaml_key in env_block:
            hyperparams_override[scratch_key] = env_block[yaml_key]

    output_dir = str(
        PROJECT_ROOT / "results" / "artifacts" / f"{config_stem}_mode{mode}_seed{seed}"
    )

    print(f"\n[train_zoo] {config_stem}  mode={mode}  seed={seed}  (custom impl: {impl_path})")
    print(f"[train_zoo] hyperparams_override: {hyperparams_override}")
    print(f"[train_zoo] output_dir: {output_dir}\n")

    try:
        module.train_model(
            task=task,
            flight_mode=mode,
            seed=seed,
            num_timesteps=num_timesteps,
            log_to_wandb=not no_wandb,
            output_dir=output_dir,
            hyperparams_override=hyperparams_override,
        )
        return 0
    except Exception as exc:
        print(f"[train_zoo] EXCEPTION: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def build_rl_zoo_cmd(
    algo: str,
    env_id: str,
    hyperparams: dict,
    mode: int,
    seed: int,
    timesteps: int | None,
    no_wandb: bool,
    config_stem: str,
    device: str,
) -> tuple[list[str], Path]:
    """Build the rl_zoo3 subprocess command and a temp YAML with flight_mode injected.

    Returns (cmd, tmp_yaml_path).  The caller is responsible for deleting the
    temp file after the subprocess exits.
    """
    # Inject flight_mode into env_kwargs so rl_zoo3 passes it to gymnasium.make
    params = copy.deepcopy(hyperparams)
    env_block = params[env_id]
    env_block.setdefault("env_kwargs", {})["flight_mode"] = mode

    # Write a temporary hyperparams YAML (rl_zoo3 needs a file path, not stdin)
    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        prefix=f"zoo_{config_stem}_m{mode}_s{seed}_",
        dir=tmp_dir,
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(params, tmp_file, default_flow_style=False, allow_unicode=True)
    tmp_file.close()
    tmp_path = Path(tmp_file.name)

    cmd = [
        sys.executable, "-m", "rl_zoo3.train",
        "--algo", algo,
        "--env", env_id,
        "-conf", str(tmp_path),
        "--seed", str(seed),
        "-f", str(PROJECT_ROOT / "results" / "zoo"),
        "--gym-packages", "PyFlyt.gym_envs",
        "--device", device,
        "--eval-episodes", "20",
        "--eval-freq", "10000",
    ]

    if timesteps is not None:
        cmd += ["-n", str(timesteps)]

    if not no_wandb:
        cmd += [
            "--track",
            "--wandb-project-name", "pyflyt-rl",
            "--wandb-group", config_stem,
            "-tags", config_stem, f"mode_{mode}", f"seed_{seed}", algo,
        ]

    return cmd, tmp_path


def run_single(
    algo: str,
    env_id: str,
    hyperparams: dict,
    config_stem: str,
    mode: int,
    seed: int,
    timesteps: int | None,
    no_wandb: bool,
    device: str,
) -> int:
    """Run one (mode, seed) training job. Returns the subprocess exit code."""
    cmd, tmp_path = build_rl_zoo_cmd(
        algo, env_id, hyperparams, mode, seed, timesteps, no_wandb, config_stem, device
    )

    # Make the project root importable so scripts.wrappers can be found by rl_zoo3
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env["WANDB_RUN_NAME"] = f"{config_stem}_seed{seed}"

    print(f"\n[train_zoo] {config_stem}  mode={mode}  seed={seed}")
    print(f"[train_zoo] cmd: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="RL Zoo 3 training wrapper for PyFlyt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "--configs", dest="config", required=True, nargs="+",
        help="One or more YAML config files (e.g. configs/sac_hover_g098_utd3.yml)",
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override n_timesteps from the config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")

    # Single run vs sweep
    single = parser.add_argument_group("Single run")
    single.add_argument("--mode", type=int, help="Flight mode")
    single.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    sweep = parser.add_argument_group("Sweep (overrides --mode / --seed)")
    sweep.add_argument("--modes", nargs="+", type=int,
                       help="Multiple flight modes to sweep")
    sweep.add_argument("--seeds", nargs="+", type=int,
                       help="Multiple seeds to sweep")

    args = parser.parse_args()

    # Resolve modes and seeds
    modes = args.modes if args.modes else ([args.mode] if args.mode is not None else None)
    seeds = args.seeds if args.seeds else [args.seed]

    if modes is None:
        parser.error("Provide --mode or --modes")

    config_paths = [Path(c) for c in args.config]
    for config_path in config_paths:
        if not config_path.exists():
            parser.error(f"Config not found: {config_path}")

    print(f"Configs:  {[str(c) for c in config_paths]}")
    print(f"Modes:    {modes}")
    print(f"Seeds:    {seeds}")
    if args.timesteps:
        print(f"Timesteps override: {args.timesteps:,}")

    failures = []
    for config_path in config_paths:
        algo, task, env_id, hyperparams, impl_path = load_config(config_path)
        config_stem = config_path.stem
        print(f"\n--- {config_stem}  ({algo} / {env_id}) ---")
        for mode, seed in itertools.product(modes, seeds):
            if impl_path:
                rc = run_single_impl(
                    impl_path, task, env_id, hyperparams, config_stem,
                    mode, seed, args.timesteps, args.no_wandb,
                )
            else:
                rc = run_single(
                    algo, env_id, hyperparams, config_stem,
                    mode, seed, args.timesteps, args.no_wandb, args.device,
                )
            if rc != 0:
                failures.append((config_stem, mode, seed, rc))
                print(f"[train_zoo] FAILED  {config_stem}  mode={mode}  seed={seed}  exit={rc}")

    if failures:
        print(f"\n[train_zoo] {len(failures)} run(s) failed:")
        for config_stem, mode, seed, rc in failures:
            print(f"  {config_stem}  mode={mode}  seed={seed}  exit={rc}")
        sys.exit(1)

    print("\n[train_zoo] All runs completed successfully.")
    print(f"Results saved to: {PROJECT_ROOT / 'results' / 'zoo'}")


if __name__ == "__main__":
    main()
