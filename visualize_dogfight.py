"""Visualize dogfight matches with rendering.

Usage:
    # Watch your trained agent vs heuristic opponent
    python visualize_dogfight.py --model submissions/group_dogfight.py --opponent heuristic

    # Watch vs another trained model
    python visualize_dogfight.py --model submissions/group_dogfight.py --opponent-model submissions/baseline_heuristic.py

    # Watch from opponent's perspective
    python visualize_dogfight.py --model submissions/group_dogfight.py --camera opponent

    # Auto-switch camera between both aircraft
    python visualize_dogfight.py --model submissions/group_dogfight.py --camera both

    # Watch vs random opponent
    python visualize_dogfight.py --model submissions/group_dogfight.py --opponent random

    # Render multiple matches back-to-back
    python visualize_dogfight.py --model submissions/group_dogfight.py --matches 3

    # Slow motion (frame delay in ms)
    python visualize_dogfight.py --model submissions/group_dogfight.py --delay 50
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


def update_camera(env, target_agent: str, camera_distance: float = 50.0, debug: bool = False):
    """Update PyBullet camera to follow target aircraft with a wide chase view.
    
    Args:
        env: The MAFixedwingDogfightEnvV2 environment
        target_agent: 'uav_0' or 'uav_1' - which aircraft to follow
        camera_distance: Distance behind aircraft (default: 50 meters for wider view)
        debug: Print debug information
    """
    try:
        import pybullet as p
        
        if debug:
            print(f"[CAMERA DEBUG] Target: {target_agent}, checking env attributes...")
        
        # Get physics client ID from environment
        physics_client = None
        if hasattr(env, '_physics_client_id'):
            physics_client = env._physics_client_id
        elif hasattr(env, 'physics_client'):
            physics_client = env.physics_client
        elif hasattr(env, 'client_id'):
            physics_client = env.client_id
        
        # Try to find aircraft through various attribute paths
        aircraft = None
        body_id = None
        
        # Method 1: agents_dict attribute
        if hasattr(env, 'agents_dict') and target_agent in env.agents_dict:
            aircraft = env.agents_dict[target_agent]
        # Method 2: _agents_dict (private)
        elif hasattr(env, '_agents_dict') and target_agent in env._agents_dict:
            aircraft = env._agents_dict[target_agent]
        # Method 3: agents list/dict
        elif hasattr(env, 'agents'):
            if isinstance(env.agents, dict) and target_agent in env.agents:
                aircraft = env.agents[target_agent]
        
        # Get body ID from aircraft object
        if aircraft is not None:
            for attr in ['id', 'body_id', 'plane_id', 'aircraft_id', 'uav_id', 'object_id']:
                if hasattr(aircraft, attr):
                    body_id = getattr(aircraft, attr)
                    break
        
        if body_id is None:
            # Fallback: try to find by name in pybullet
            if physics_client is not None:
                num_bodies = p.getNumBodies(physicsClientId=physics_client)
            else:
                num_bodies = p.getNumBodies()
            
            for i in range(num_bodies):
                try:
                    if physics_client is not None:
                        info = p.getBodyInfo(i, physicsClientId=physics_client)
                    else:
                        info = p.getBodyInfo(i)
                    # Check if body name contains the agent name
                    if target_agent.encode() in info[1] or b'uav' in info[1].lower():
                        body_id = i
                        break
                except:
                    continue
        
        if body_id is None:
            return
        
        # Get position and orientation
        if physics_client is not None:
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
        else:
            pos, orn = p.getBasePositionAndOrientation(body_id)
        
        # Get rotation matrix from quaternion
        if physics_client is not None:
            rot_matrix = p.getMatrixFromQuaternion(orn, physicsClientId=physics_client)
        else:
            rot_matrix = p.getMatrixFromQuaternion(orn)
        
        # Forward vector (aircraft's forward direction) - correct for aircraft frame
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        
        # Up vector
        up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
        
        # Camera position: far behind and above
        cam_distance = camera_distance
        height_offset = 20.0
        
        cam_pos = [
            pos[0] - forward[0] * cam_distance,
            pos[1] - forward[1] * cam_distance,
            pos[2] + height_offset
        ]
        
        # Camera target: look at aircraft position
        cam_target = list(pos)
        
        # Calculate yaw (angle around Z axis)
        dx = pos[0] - cam_pos[0]
        dy = pos[1] - cam_pos[1]
        yaw = np.degrees(np.arctan2(dy, dx))
        
        # Calculate pitch (angle up/down)
        horizontal_dist = np.sqrt(dx*dx + dy*dy)
        dz = pos[2] - cam_pos[2]
        pitch = np.degrees(np.arctan2(dz, horizontal_dist))
        
        # Use resetDebugVisualizerCamera with explicit physics client
        if physics_client is not None:
            # For GUI mode, we need to use the default connection
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_distance,
                cameraYaw=yaw,
                cameraPitch=pitch - 10,  # Slight downward tilt
                cameraTargetPosition=cam_target,
                physicsClientId=physics_client
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_distance,
                cameraYaw=yaw,
                cameraPitch=pitch - 10,
                cameraTargetPosition=cam_target
            )
            
    except Exception as e:
        if debug:
            print(f"[CAMERA DEBUG] Error: {e}")
        # Silently fail - camera not critical for visualization
        pass


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
                         seed: int = 0, delay_ms: int = 0, max_steps: int = 1800,
                         camera_mode: str = "agent", debug: bool = False):
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
    print(f"Seed: {seed} | Camera: {camera_mode}")
    print(f"{'='*60}")
    
    if debug:
        print(f"[DEBUG] Env attributes: {[a for a in dir(env) if not a.startswith('_')][:20]}...")
        if hasattr(env, 'agents_dict'):
            print(f"[DEBUG] agents_dict keys: {list(env.agents_dict.keys()) if env.agents_dict else 'None'}")
        if hasattr(env, '_physics_client_id'):
            print(f"[DEBUG] physics_client_id: {env._physics_client_id}")
    
    # Camera switching state
    current_camera_target = "uav_0" if camera_mode in ["agent", "both"] else "uav_1"
    last_camera_switch = 0
    camera_switch_interval = 180  # frames (~6 seconds at 30Hz)

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

        # Handle camera switching for 'both' mode
        if camera_mode == "both":
            if step - last_camera_switch > camera_switch_interval:
                current_camera_target = "uav_1" if current_camera_target == "uav_0" else "uav_0"
                last_camera_switch = step
                print(f"  [Camera] Now following: {current_camera_target}")
        elif camera_mode == "agent":
            current_camera_target = "uav_0"
        elif camera_mode == "opponent":
            current_camera_target = "uav_1"
        
        # Update camera position (every 10 frames to avoid jitter)
        if camera_mode != "overview" and step % 10 == 0:
            update_camera(env, current_camera_target, debug=debug)

        if delay_ms > 0:
            import time
            time.sleep(delay_ms / 1000.0)

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
    parser.add_argument("--delay", type=int, default=0,
                        help="Frame delay in milliseconds (slow motion, default: 0)")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed for matches (default: 0)")
    parser.add_argument("--camera", type=str, default="agent",
                        choices=["agent", "opponent", "both", "overview"],
                        help="Camera view: agent (your model), opponent, both (switches), or overview (default: agent)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output for camera troubleshooting")
    args = parser.parse_args()

    results = []
    for i in range(args.matches):
        result, r_our, r_opp = run_visualized_match(
            args.model,
            opponent_type=args.opponent,
            opponent_model_path=args.opponent_model,
            seed=args.seed_start + i,
            delay_ms=args.delay,
            camera_mode=args.camera,
            debug=args.debug,
        )
        results.append((result, r_our, r_opp))

    if args.matches > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {args.matches} matches vs {args.opponent}")
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
