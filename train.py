"""Generic training entrypoint for any model implementation module.

Single run:
    python train.py --model implementations/ppo_lib.py --task waypoints --mode 6 --seed 0 --timesteps 20000 --no-wandb

Single run across multiple implementations:
    python train.py --model implementations/ppo_lib.py implementations/ppo_implementation.py --task hover --mode 0 --seed 0 --timesteps 20000 --no-wandb

Full benchmark across tasks, modes, and seeds:
    python train.py --model implementations/ppo_lib.py --benchmark --modes -1 0 4 6 7 --seeds 0 1 2 --timesteps 100000 --no-wandb

The benchmark runs every (task, mode, seed) combination and prints a summary table at the end.
"""

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress TF C++ info at SB3 import time

import argparse
import importlib.util
import itertools
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from implementations.model_utils import finish_wandb, init_wandb_run


PROJECT_ROOT = Path(__file__).resolve().parent

# Default benchmark configuration
BENCHMARK_TASKS = ["hover", "waypoints"]
BENCHMARK_MODES = [0, 4, 6, 7]
BENCHMARK_SEEDS = [0, 1, 2]


def import_module_from_path(module_path, module_name="train_target"):
    """Import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_name(algorithm_name, task, mode, seed):
    """Return run name used for results"""
    return f"{algorithm_name}_{task}_mode{mode}_seed{seed}"


def default_artifact_dir(algorithm_name, mode, seed, task):
    """Return the default artifact path (.pt/.json files)."""
    return PROJECT_ROOT / "results" / "artifacts" / run_name(algorithm_name, task, mode, seed)


def default_submission_module_path(algorithm_name, mode, seed, task):
    """Return the generated submission .py path used by scripts/evaluate.py."""
    return PROJECT_ROOT / "results" / "models" / f"{run_name(algorithm_name, task, mode, seed)}.py"


def write_submission_module(submission_path, artifact_dir, model_info):
    """Generate a .py module exposing load_model() for evaluate.py."""
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    implementation_rel = model_info["implementation_path"]
    algorithm_name = model_info.get("algorithm", "model")
    task = model_info.get("task", "hover")

    content = f'''"""Auto-generated evaluation entrypoint.

Usage:
    python scripts/evaluate.py --model {submission_path.as_posix()} --env {task}
"""

from pathlib import Path
import importlib.util

ALGORITHM_NAME = "{algorithm_name}"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "{Path(artifact_dir).name}"
IMPLEMENTATION_PATH = Path(__file__).resolve().parent.parent.parent / "{implementation_rel}"


def _import_impl():
    spec = importlib.util.spec_from_file_location("submission_impl", str(IMPLEMENTATION_PATH))
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import implementation: {{IMPLEMENTATION_PATH}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(path=None):
    mod = _import_impl()
    target = Path(path) if path else ARTIFACT_DIR
    return mod.load_model(str(target))
'''

    with open(submission_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_single(module, module_path, algorithm_name, task, mode, seed, timesteps, no_wandb, output_dir=None):
    """Train one (task, mode, seed) configuration. Returns (artifact_dir, model_info, error)."""
    out = Path(output_dir) if output_dir else default_artifact_dir(algorithm_name, mode, seed, task)
    out.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        summary = module.train_model(
            task=task,
            flight_mode=mode,
            seed=seed,
            num_timesteps=timesteps,
            log_to_wandb=not no_wandb,
            output_dir=str(out),
        )
        elapsed = time.time() - t0
    except Exception as exc:
        return out, None, exc

    model_info = {
        "algorithm": algorithm_name,
        "implementation_path": os.path.relpath(module_path, PROJECT_ROOT).replace("\\", "/"),
        "checkpoint_file": "checkpoint.pt",
        "entrypoint": "load_model",
        "task": task,
        "flight_mode": mode,
        "seed": seed,
        "num_timesteps": timesteps,
        "training_time_s": round(elapsed, 1),
    }
    if isinstance(summary, dict):
        model_info.update(summary)

    submission_path = default_submission_module_path(algorithm_name, mode, seed, task)
    model_info["artifact_dir"] = os.path.relpath(out, PROJECT_ROOT).replace("\\", "/")
    model_info["submission_file"] = os.path.relpath(submission_path, PROJECT_ROOT).replace("\\", "/")

    with open(out / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    write_submission_module(submission_path, out, model_info)
    return out, model_info, None


def aggregate_benchmark_results(results):
    grouped = {}
    for result in results:
        key = (result["task"], result["mode"])
        if key not in grouped:
            grouped[key] = {
                "total": 0,
                "ok": 0,
                "times": [],
                "eval_rewards": [],
            }

        bucket = grouped[key]
        bucket["total"] += 1
        if result["status"] == "OK":
            bucket["ok"] += 1
            if isinstance(result.get("time_s"), (int, float)):
                bucket["times"].append(float(result["time_s"]))
            eval_reward = result.get("eval_reward_mean")
            if isinstance(eval_reward, (int, float)):
                bucket["eval_rewards"].append(float(eval_reward))

    rows = []
    for task, mode in sorted(grouped.keys()):
        bucket = grouped[(task, mode)]
        rows.append({
            "task": task,
            "mode": mode,
            "mean_time_s": statistics.mean(bucket["times"]) if bucket["times"] else None,
            "mean_eval_reward": statistics.mean(bucket["eval_rewards"]) if bucket["eval_rewards"] else None,
            "success_rate": 100.0 * bucket["ok"] / bucket["total"] if bucket["total"] else 0.0,
        })
    return rows


def log_benchmark_aggregates_to_wandb(enabled, algorithm_name, tasks, modes, seeds, timesteps, aggregate_rows):
    if not enabled:
        return

    init_wandb_run(
        enabled,
        project="pyflyt-rl",
        name=f"{algorithm_name}_benchmark_aggregate",
        job_type="benchmark_aggregate",
        config={
            "algorithm": algorithm_name,
            "tasks": list(tasks),
            "modes": list(modes),
            "seed_count": len(seeds),
            "timesteps": timesteps,
        },
        tags=[algorithm_name, "benchmark", "aggregate"],
        sync_tensorboard=False,
    )

    import wandb

    aggregate_table = wandb.Table(
        columns=["task", "mode", "mean_time_s", "mean_eval_reward", "success_rate"],
        data=[
            [row["task"], row["mode"], row["mean_time_s"], row["mean_eval_reward"], row["success_rate"]]
            for row in aggregate_rows
        ],
    )
    wandb.log({"benchmark_seed_aggregate": aggregate_table})

    for task in sorted({row["task"] for row in aggregate_rows}):
        task_rows = [row for row in aggregate_rows if row["task"] == task]

        time_table = wandb.Table(
            columns=["mode", "mean_time_s"],
            data=[[row["mode"], row["mean_time_s"]] for row in task_rows if row["mean_time_s"] is not None],
        )
        if len(time_table.data) > 0:
            wandb.log({
                f"{task}/mean_time_s_by_mode": wandb.plot.line(
                    time_table,
                    "mode",
                    "mean_time_s",
                    title=f"{task}: mean training time across seeds",
                )
            })

        reward_table = wandb.Table(
            columns=["mode", "mean_eval_reward"],
            data=[[row["mode"], row["mean_eval_reward"]] for row in task_rows if row["mean_eval_reward"] is not None],
        )
        if len(reward_table.data) > 0:
            wandb.log({
                f"{task}/mean_eval_reward_by_mode": wandb.plot.scatter(
                    reward_table,
                    "mode",
                    "mean_eval_reward",
                    title=f"{task}: mean eval reward across seeds",
                )
            })

        success_table = wandb.Table(
            columns=["mode", "success_rate"],
            data=[[row["mode"], row["success_rate"]] for row in task_rows],
        )
        wandb.log({
            f"{task}/success_rate_by_mode": wandb.plot.line(
                success_table,
                "mode",
                "success_rate",
                title=f"{task}: success rate across seeds",
            )
        })

    finish_wandb(enabled)


def main():
    parser = argparse.ArgumentParser(description="Train any model implementation module")
    parser.add_argument("--model", required=True, nargs="+", help="One or more paths to model implementation Python files")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Single-run options
    single = parser.add_argument_group("Single run (required when not using --benchmark)")
    single.add_argument("--task", choices=["hover", "waypoints"])
    single.add_argument("--mode", type=int, choices=[-1, 0, 4, 6, 7])
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--output", type=str, default=None, help="Override artifact directory")

    # Benchmark options
    bench = parser.add_argument_group("Benchmark (sweep over tasks, modes, and seeds)")
    bench.add_argument("--benchmark", action="store_true", help="Run full benchmark sweep")
    bench.add_argument("--tasks", nargs="+", choices=["hover", "waypoints"],
                       default=BENCHMARK_TASKS, help=f"Tasks to include (default: {BENCHMARK_TASKS})")
    bench.add_argument("--modes", nargs="+", type=int,
                       default=BENCHMARK_MODES, help=f"Flight modes to sweep (default: {BENCHMARK_MODES})")
    bench.add_argument("--seeds", nargs="+", type=int,
                       default=BENCHMARK_SEEDS, help=f"Seeds to sweep (default: {BENCHMARK_SEEDS})")

    args = parser.parse_args()

    if not args.benchmark and (args.task is None or args.mode is None):
        parser.error("--task and --mode are required for a single run (or use --benchmark)")

    if len(args.model) > 1 and args.output is not None:
        parser.error("--output can only be used when a single --model path is provided")

    model_targets = []
    for model_path_str in args.model:
        module_path = Path(model_path_str)
        if not module_path.exists():
            raise FileNotFoundError(f"Model implementation not found: {module_path}")

        module = import_module_from_path(str(module_path), module_name=f"train_target_{module_path.stem}")
        if not hasattr(module, "train_model"):
            raise ValueError(
                f"{module_path} must define train_model(task, flight_mode, seed, num_timesteps, log_to_wandb, output_dir)"
            )

        algorithm_name = getattr(module, "ALGORITHM_NAME", module_path.stem.replace("_implementation", ""))
        model_targets.append((module_path, module, algorithm_name))

    # Single run
    if not args.benchmark:
        for module_path, module, algorithm_name in model_targets:
            print(f"\nTRAINING  {module_path}  |  {args.task}  mode={args.mode}  seed={args.seed}\n")

            out, info, err = run_single(
                module, module_path, algorithm_name,
                args.task, args.mode, args.seed, args.timesteps, args.no_wandb,
                output_dir=args.output,
            )
            if err:
                print(f"ERROR: {err}")
                traceback.print_exc()
                sys.exit(1)

            print(f"Saved model artifacts to: {out}")
            print(f"Generated evaluation module: {info.get('submission_file')}")
            print(
                "Evaluate with: "
                f"python scripts/evaluate.py --model {info.get('submission_file')} --env {args.task}"
            )
        return

    configs = list(itertools.product(args.tasks, args.modes, args.seeds))
    total = len(configs)

    for module_path, module, algorithm_name in model_targets:
        results = []

        # Benchmark configuration summary
        print(f"\nBENCHMARK  {module_path}  ({total} runs)")
        print(f"  Tasks:  {args.tasks}")
        print(f"  Modes:  {args.modes}")
        print(f"  Seeds:  {args.seeds}")
        print(f"  Steps:  {args.timesteps:,}\n")

        for idx, (task, mode, seed) in enumerate(configs, 1):
            print(f"\n[{idx}/{total}] task={task}  mode={mode}  seed={seed}")
            out, info, err = run_single(
                module, module_path, algorithm_name,
                task, mode, seed, args.timesteps, args.no_wandb,
            )
            if err:
                print(f"  FAILED: {err}")
                results.append({"task": task, "mode": mode, "seed": seed, "status": "FAILED", "error": str(err)})
            else:
                print(f"  OK  →  artifact: {out}")
                print(f"        submission: {info.get('submission_file')}")
                results.append({
                    "task": task, "mode": mode, "seed": seed,
                    "status": "OK",
                    "artifact": str(out),
                    "submission": info.get("submission_file"),
                    "time_s": info.get("training_time_s"),
                    "eval_reward_mean": info.get("eval_reward_mean"),
                })

        # Print summary table
        print(f"\nBENCHMARK SUMMARY  —  {algorithm_name}\n")
        col = "{:<16} {:>6} {:>6}  {:<8}  {}"
        print(col.format("task", "mode", "seed", "status", "artifact / error"))
        print("-" * 80)
        for r in results:
            detail = r.get("submission") or r.get("artifact") or r.get("error", "")
            print(col.format(r["task"], r["mode"], r["seed"], r["status"], detail))
        print("")

        aggregate_rows = aggregate_benchmark_results(results)
        log_benchmark_aggregates_to_wandb(
            enabled=not args.no_wandb,
            algorithm_name=algorithm_name,
            tasks=args.tasks,
            modes=args.modes,
            seeds=args.seeds,
            timesteps=args.timesteps,
            aggregate_rows=aggregate_rows,
        )

if __name__ == "__main__":
    main()
