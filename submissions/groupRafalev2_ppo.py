"""Tournament submission — dogfight PPO (self-play trained)."""

def load_model():
    """Load the trained dogfight PPO model."""
    from stable_baselines3 import PPO
    return PPO.load("groupRafalev2_ppo.zip")
