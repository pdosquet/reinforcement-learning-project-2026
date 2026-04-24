"""Visualize dogfight matches with rendering.

Usage:
    # Watch your trained agent vs heuristic opponent
    python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent heuristic

    # Watch vs another trained model
    python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent-model submissions/baseline_heuristic.py

    # Watch vs random opponent
    python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent random

    # Run multiple matches back-to-back
    python visualize_dogfight.py --model submissions/groupRafalev2_ppo.py --opponent heuristic --matches 5

Camera controls (in PyBullet window):
    - Scroll wheel: Zoom in/out
    - Right-click + drag: Rotate camera
    - Middle-click + drag: Pan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2


def load_model(model_path: str):
    """Load a model from .py submission or .zip checkpoint."""
    if model_path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_model()
    # Legacy SB3 .zip fallback
    from stable_baselines3 import PPO, SAC
    for cls in [PPO, SAC]:
        try:
            return cls.load(model_path)
        except Exception:
            continue
    raise ValueError(f"Could not load model: {model_path}")


def get_opponent_action(env, agent: str, opponent_type: str):
    """Get opponent action."""
    if opponent_type == "random":
        return env.action_space(agent).sample()
    else:  # heuristic
        act = np.zeros(4, dtype=np.float32)
        act[0] = 0.0   # roll
        act[1] = 0.05  # slight pitch up
        act[2] = 0.0   # yaw
        act[3] = 0.65  # throttle
        return act


def run_visualized_match(model_path: str, opponent_type: str = "heuristic",
                         opponent_model_path: str | None = None,
                         seed: int = 0, max_steps: int = 1800):
    """Run a single visualized dogfight match."""
    model = load_model(model_path)
    
    # Load opponent model if provided
    opponent_model = None
    if opponent_model_path:
        opponent_model = load_model(opponent_model_path)
        opponent_type = "model"

    env = MAFixedwingDogfightEnvV2(
        team_size=1,
        assisted_flight=True,
        flatten_observation=True,
        render_mode="human",
        max_duration_seconds=60,
        agent_hz=30,
    )

    observations, infos = env.reset(seed=seed)
    rewards_acc = {"uav_0": 0.0, "uav_1": 0.0}

    opp_name = opponent_model_path if opponent_model_path else opponent_type
    print(f"\n{'='*60}")
    print(f"Match starting: {Path(model_path).name} vs {Path(opp_name).name if '/' in opp_name or '\\' in opp_name else opp_name}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    for step in range(max_steps):
        actions = {}
        for agent in env.agents:
            if agent == "uav_0":
                actions[agent] = model.predict(observations[agent], deterministic=True)[0]
            elif opponent_model is not None:
                # Use loaded opponent model
                actions[agent] = opponent_model.predict(observations[agent], deterministic=True)[0]
            else:
                actions[agent] = get_opponent_action(env, agent, opponent_type)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            rewards_acc[agent] += rewards.get(agent, 0.0)

        all_done = all(
            terminations.get(a, True) or truncations.get(a, False)
            for a in env.agents
        ) or len(env.agents) == 0

        if all_done:
            break

    env.close()

    # Determine winner
    r_our = rewards_acc.get("uav_0", 0.0)
    r_opp = rewards_acc.get("uav_1", 0.0)

    a_won, b_won = False, False
    for agent, info in infos.items():
        if info.get("team_win", False):
            if agent == "uav_0":
                a_won = True
            else:
                b_won = True

    if not a_won and not b_won:
        if r_our > r_opp + 50:
            a_won = True
        elif r_opp > r_our + 50:
            b_won = True

    if a_won and not b_won:
        result = "WIN"
    elif b_won and not a_won:
        result = "LOSS"
    else:
        result = "DRAW"

    print(f"\nResult: {result}")
    print(f"Your agent reward: {r_our:.1f}")
    print(f"Opponent reward:   {r_opp:.1f}")
    print(f"Steps: {step + 1}")
    print(f"{'='*60}\n")

    return result, r_our, r_opp


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dogfight matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to your trained model (.py or .zip)")
    parser.add_argument("--opponent", type=str, default="heuristic",
                        choices=["heuristic", "random"],
                        help="Built-in opponent type (default: heuristic)")
    parser.add_argument("--opponent-model", type=str, default=None,
                        help="Path to opponent model file (.py or .zip). Overrides --opponent if set.")
    parser.add_argument("--matches", type=int, default=1,
                        help="Number of matches to run (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for first match (default: 0)")
    args = parser.parse_args()

    results = []
    for i in range(args.matches):
        result, r_our, r_opp = run_visualized_match(
            args.model,
            opponent_type=args.opponent,
            opponent_model_path=args.opponent_model,
            seed=args.seed + i,
        )
        results.append((result, r_our, r_opp))
    
    if args.matches > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {args.matches} matches")
        print(f"{'='*60}")
        wins = sum(1 for r, _, _ in results if r == "WIN")
        losses = sum(1 for r, _, _ in results if r == "LOSS")
        draws = sum(1 for r, _, _ in results if r == "DRAW")
        print(f"  Wins:   {wins} ({100*wins/args.matches:.1f}%)")
        print(f"  Losses: {losses} ({100*losses/args.matches:.1f}%)")
        print(f"  Draws:  {draws} ({100*draws/args.matches:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
