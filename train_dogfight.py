"""Self-play training script for dogfight PPO using YAML configs.

Trains a PPO agent in DogfightSelfPlayEnv and writes:
  - results/dogfight/dogfight_ppo_seed{seed}.zip      (SB3 checkpoint)
  - results/dogfight/dogfight_ppo_seed{seed}_meta.json

Usage:
    # Basic run with default config
    python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0

    # No W&B logging
    python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0 --no-wandb

    # Multiple seeds
    python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seeds 0 1 2

    # Override timesteps from command line
    python train_dogfight.py --config configs/dogfight/dogfight_ppo_default.yml --seed 0 --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from implementations.dogfight_ppo import train_selfplay


def load_config(config_path: Path) -> dict:
    """Load dogfight hyperparameters from YAML config."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw.get("dogfight", {})

def main():
    parser = argparse.ArgumentParser(
        description="Self-play PPO training for the dogfight tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g., configs/dogfight/dogfight_ppo.yml)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps from config")
    parser.add_argument("--seed", type=int, default=0,
                        help="Single random seed (use --seeds for multiple)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Multiple seeds (overrides --seed)")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device: cpu or cuda (default: cpu)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output-dir", default="results/dogfight",
                        help="Output directory for checkpoints (default: results/dogfight/)")
    args = parser.parse_args()

    # Load hyperparameters from YAML
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = args.config

    hyperparams = load_config(config_path)

    # Override timesteps if specified
    if args.timesteps is not None:
        hyperparams["n_timesteps"] = args.timesteps

    # Get timesteps from config (default to 500k)
    num_timesteps = hyperparams.get("n_timesteps", 500_000)

    seeds = args.seeds if args.seeds else [args.seed]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Training dogfight agent  seed={seed}  steps={num_timesteps:,}")
        print(f"  Config: {args.config}")
        print(f"{'='*60}")
        checkpoint_path = train_selfplay(
            seed=seed,
            num_timesteps=num_timesteps,
            log_to_wandb=not args.no_wandb,
            output_dir=args.output_dir,
            device=args.device,
            hyperparams=hyperparams,
        )
        print(f"[train_dogfight] Checkpoint saved → {checkpoint_path}")

    print("\n[train_dogfight] Done.")
    print(f"  Checkpoints: results/dogfight/dogfight_ppo_seed*.zip")


if __name__ == "__main__":
    main()
