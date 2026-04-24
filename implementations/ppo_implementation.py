"""Scratch PPO model implementation usable by train.py."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# Task episode limits for evaluation
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
        return gymnasium.make(
            "PyFlyt/QuadX-Waypoints-v4",
            flight_mode=flight_mode,
            render_mode=render_mode,
        )
    raise ValueError(f"Unknown task: {task}")


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


def force_flight_mode(flight_mode: int):
    """Patch gymnasium.make so evaluate.py creates the environment with the saved mode."""
    original_make = gymnasium.make

    def patched_make(env_id, *args, **kwargs):
        kwargs["flight_mode"] = flight_mode
        return original_make(env_id, *args, **kwargs)

    gymnasium.make = patched_make


def load_artifact_metadata(artifact_dir: Path):
    """Load artifact metadata from JSON file if it exists."""
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_checkpoint_path(artifact_dir: Path, metadata: dict, default_name: str = "checkpoint.pt"):
    """Resolve checkpoint path, preferring 'best_checkpoint.pt' if it exists."""
    path = artifact_dir
    best_checkpoint = path / "best_checkpoint.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    return path / metadata.get("checkpoint_file", default_name)


# W&B logging utilities (compatible with train_zoo.py configuration)
def init_wandb_run(enabled: bool, **kwargs):
    """Initialize a W&B run only when logging is enabled."""
    if not enabled:
        return None
    import os
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


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, action_std_init=0.3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        # Reduced initial action std from 0.6 to 0.3 for better stability
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std).expand(state.shape[0], -1)
        return mean, std

    def get_distribution(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        return self.fc_value(x)


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device="cpu",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        hidden_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic = CriticNetwork(state_dim, hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        # Use lower learning rate for critic for better stability
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate * 0.5)
        
        # Learning rate schedulers for gradual decay
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000, eta_min=learning_rate * 0.1
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=1000, eta_min=learning_rate * 0.05
        )

    def predict(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_distribution(obs)
            action = dist.mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu().numpy(), {"log_prob": log_prob.item()}

    def compute_advantages(self, states, rewards, values, dones, next_value):
        advantages = np.zeros(len(rewards))
        gae = 0.0

        for index in reversed(range(len(rewards))):
            if index == len(rewards) - 1:
                next_val = next_value
                next_done = 0
            else:
                next_val = values[index + 1]
                next_done = dones[index + 1]

            delta = rewards[index] + self.gamma * next_val * (1.0 - next_done) - values[index]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[index] = gae

        returns = advantages + values
        # Clip extreme advantages to prevent training instability
        advantages = np.clip(advantages, -5.0, 5.0)
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns, epochs=10):
        # Tighter normalization and clipping for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -3.0, 3.0)  # Tighter clipping: ±3 instead of ±10
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for value function fitting
        returns_normalized = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        actor_losses = []
        critic_losses = []
        clip_fractions = []
        policy_kl_divs = []

        for epoch_idx in range(epochs):
            dist = self.actor.get_distribution(states_t)
            log_probs_new = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().mean()
            values_new = self.critic(states_t).squeeze(-1)

            ratio = torch.exp(log_probs_new - log_probs_old_t)
            
            # Clipped surrogate loss with smaller clipping range for stability
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
            
            # Entropy decays over epochs within update
            entropy_coef_scaled = self.entropy_coef * (1.0 - epoch_idx / max(1, epochs))
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef_scaled * entropy
            
            # Improved value loss with Huber-like clipping per sample
            value_error = values_new - returns_normalized
            # Clip large value errors to prevent value divergence
            value_error_clipped = torch.clamp(value_error, -1.0, 1.0)
            critic_loss = (value_error_clipped ** 2).mean() * self.value_loss_coef

            with torch.no_grad():
                clipped = (ratio > 1 + self.clip_ratio) | (ratio < 1 - self.clip_ratio)
                clip_fractions.append(clipped.float().mean().item())
                # Track KL divergence for debugging policy drift
                kl_div = (log_probs_old_t - log_probs_new).mean().item()
                policy_kl_divs.append(kl_div)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # More conservative gradient clipping
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm * 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # More conservative gradient clipping for value network
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm * 0.5)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        # Step learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(entropy.item()),
            "clip_fraction": float(np.mean(clip_fractions)),
            "policy_kl_div": float(np.mean(policy_kl_divs)),
        }

    def save(self, filepath, hyperparams=None):
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "hyperparams": hyperparams or {},
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


class PPOTrainer:
    def __init__(self, env, agent, trajectory_length=2048, epochs_per_update=10):
        self.env = env
        self.agent = agent
        self.trajectory_length = trajectory_length
        self.epochs_per_update = epochs_per_update
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def collect_trajectory(self, num_steps):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        obs, _ = self.env.reset()

        for _ in range(num_steps):
            states.append(obs.copy())
            action, info = self.agent.predict(obs, deterministic=False)
            actions.append(action)
            log_probs.append(info["log_prob"])

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                values.append(self.agent.critic(obs_t).item())

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            dones.append(done)

            if done:
                obs, _ = self.env.reset()

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
            next_value = self.agent.critic(obs_t).item()

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "dones": np.array(dones),
            "next_value": next_value,
        }


def load_checkpoint_metadata(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dim" not in checkpoint or "action_dim" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing state_dim/action_dim metadata. "
            "Re-train with the current training pipeline."
        )
    return checkpoint


ALGORITHM_NAME = "ppo"
PPO_HYPERPARAMS = {
    "learning_rate": 2e-4,  # Reduced from 3e-4 for more conservative updates
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_size": 64,
    "trajectory_length": 2048,
    "epochs_per_update": 10,
}


def train_model(
    task,
    flight_mode,
    seed,
    num_timesteps,
    log_to_wandb,
    output_dir,
    eval_interval=5,
    quick_eval_episodes=5,
    hyperparams_override=None,
):
    """Train the scratch PPO agent and save checkpoints in output_dir.

    Args:
        hyperparams_override: optional dict that overrides individual keys in
            PPO_HYPERPARAMS (e.g. when called from train_zoo.py with a YAML config).
    """
    hp = {**PPO_HYPERPARAMS}
    if hyperparams_override:
        hp.update({k: v for k, v in hyperparams_override.items() if k in PPO_HYPERPARAMS})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = output_path / "best_checkpoint.pt"
    checkpoint_path = output_path / "checkpoint.pt"
    best_eval_reward = -float("inf")

    env = create_env(task, flight_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_wandb_run(
        log_to_wandb,
        project="pyflyt-rl",
        name=f"{ALGORITHM_NAME}_{task}_mode{flight_mode}_seed{seed}",
        config={
            "algorithm": ALGORITHM_NAME,
            "task": task,
            **hp,
        },
        tags=[ALGORITHM_NAME, task, f"mode_{flight_mode}"],
        sync_tensorboard=False,
    )

    try:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            clip_ratio=hp["clip_ratio"],
            entropy_coef=hp["entropy_coef"],
            value_loss_coef=hp["value_loss_coef"],
            max_grad_norm=hp["max_grad_norm"],
            hidden_size=hp["hidden_size"],
        )
        trainer = PPOTrainer(
            env=env,
            agent=agent,
            trajectory_length=hp["trajectory_length"],
            epochs_per_update=hp["epochs_per_update"],
        )

        total_timesteps = 0
        update_count = 0

        while total_timesteps < num_timesteps:
            trajectory = trainer.collect_trajectory(hp["trajectory_length"])
            total_timesteps += hp["trajectory_length"]

            advantages, returns = agent.compute_advantages(
                trajectory["states"],
                trajectory["rewards"],
                trajectory["values"],
                trajectory["dones"],
                trajectory["next_value"],
            )
            update_metrics = agent.update(
                trajectory["states"],
                trajectory["actions"],
                trajectory["log_probs"],
                advantages,
                returns,
                epochs=hp["epochs_per_update"],
            )
            update_count += 1

            if log_to_wandb:
                log_wandb(
                    log_to_wandb,
                    {
                        "actor_loss": update_metrics["actor_loss"],
                        "critic_loss": update_metrics["critic_loss"],
                        "entropy": update_metrics["entropy"],
                        "clip_fraction": update_metrics["clip_fraction"],
                        "policy_kl_div": update_metrics["policy_kl_div"],
                    },
                    step=total_timesteps,
                )

            if update_count % eval_interval == 0:
                stats = evaluate_policy(
                    agent,
                    env,
                    TASK_EPISODE_LIMITS[task],
                    n_episodes=quick_eval_episodes,
                )
                if stats["mean_reward"] > best_eval_reward:
                    best_eval_reward = stats["mean_reward"]
                    agent.save(str(best_checkpoint_path), hyperparams=hp)
                print(
                    f"  [update {update_count:4d} | steps {total_timesteps:>10,}] "
                    f"eval={stats['mean_reward']:.2f}  best={best_eval_reward:.2f}"
                )
                if log_to_wandb:
                    log_wandb(
                        log_to_wandb,
                        {
                            "eval_mean_reward": stats["mean_reward"],
                            "best_eval_reward": best_eval_reward,
                        },
                        step=total_timesteps,
                    )

        agent.save(str(checkpoint_path), hyperparams=hp)
        checkpoint_name = "best_checkpoint.pt" if best_checkpoint_path.exists() else checkpoint_path.name

        summary = {
            "algorithm": ALGORITHM_NAME,
            "task": task,
            "flight_mode": flight_mode,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "checkpoint_file": checkpoint_name,
            "eval_reward_mean": best_eval_reward if best_checkpoint_path.exists() else float("nan"),
        }

        wandb_summary = {
            "checkpoint_file": checkpoint_name,
            "eval_reward_mean": summary["eval_reward_mean"],
        }
        finish_wandb(log_to_wandb, wandb_summary)

        env.close()
        return summary
    except Exception:
        finish_wandb(log_to_wandb)
        env.close()
        raise


def load_model(path=None):
    """Load a scratch PPO checkpoint from an artifact directory or file."""
    if path is None:
        raise ValueError("A checkpoint file or artifact directory path is required")

    resolved_path = Path(path)
    metadata = {}

    if resolved_path.is_dir():
        metadata = load_artifact_metadata(resolved_path)
        checkpoint_path = resolve_checkpoint_path(resolved_path, metadata, default_name="checkpoint.pt")
        if "flight_mode" in metadata:
            force_flight_mode(metadata["flight_mode"])
    else:
        checkpoint_path = resolved_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint_metadata(str(checkpoint_path), device)
    hyperparams = checkpoint.get("hyperparams", {})

    agent = PPOAgent(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        device=device,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        gamma=hyperparams.get("gamma", 0.99),
        gae_lambda=hyperparams.get("gae_lambda", 0.95),
        clip_ratio=hyperparams.get("clip_ratio", 0.2),
        entropy_coef=hyperparams.get("entropy_coef", 0.01),
        value_loss_coef=hyperparams.get("value_loss_coef", 0.5),
        max_grad_norm=hyperparams.get("max_grad_norm", 0.5),
        hidden_size=hyperparams.get("hidden_size", 64),
    )
    agent.load(str(checkpoint_path))

    return ObservationCheckedModel(agent, expected_obs_dim=agent.state_dim, task=metadata.get("task"))
