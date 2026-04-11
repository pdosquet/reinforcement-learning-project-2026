"""Scratch PPO model implementation usable by train.py."""

from __future__ import annotations

from collections import deque
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.model_utils import (
    TASK_EPISODE_LIMITS,
    ObservationCheckedModel,
    create_env,
    evaluate_policy,
    finish_wandb,
    force_flight_mode,
    init_wandb_run,
    log_wandb,
    load_artifact_metadata,
    resolve_checkpoint_path,
)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, action_std_init=0.6):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
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

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic = CriticNetwork(state_dim, hidden_size).to(device)
        self.actor_old = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self._copy_actor_to_old()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate / 2)

    def _copy_actor_to_old(self):
        self.actor_old.load_state_dict(self.actor.state_dict())

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
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns, epochs=10):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        actor_losses = []
        critic_losses = []
        clip_fractions = []

        for _ in range(epochs):
            dist = self.actor.get_distribution(states_t)
            log_probs_new = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().mean()
            values_new = self.critic(states_t).squeeze(-1)

            ratio = torch.exp(log_probs_new - log_probs_old_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = ((values_new - returns_t) ** 2).mean()

            with torch.no_grad():
                clipped = (ratio > 1 + self.clip_ratio) | (ratio < 1 - self.clip_ratio)
                clip_fractions.append(clipped.float().mean().item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(entropy.item()),
            "clip_fraction": float(np.mean(clip_fractions)),
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
        self._copy_actor_to_old()


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
    "learning_rate": 3e-4,
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
    eval_interval=10,
    quick_eval_episodes=5,
):
    """Train the scratch PPO agent and save checkpoints in output_dir."""
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
            "flight_mode": flight_mode,
            "seed": seed,
            "num_timesteps": num_timesteps,
            **PPO_HYPERPARAMS,
        },
        tags=[ALGORITHM_NAME, task, f"mode_{flight_mode}"],
        sync_tensorboard=False,
    )

    try:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=PPO_HYPERPARAMS["learning_rate"],
            gamma=PPO_HYPERPARAMS["gamma"],
            gae_lambda=PPO_HYPERPARAMS["gae_lambda"],
            clip_ratio=PPO_HYPERPARAMS["clip_ratio"],
            entropy_coef=PPO_HYPERPARAMS["entropy_coef"],
            value_loss_coef=PPO_HYPERPARAMS["value_loss_coef"],
            max_grad_norm=PPO_HYPERPARAMS["max_grad_norm"],
            hidden_size=PPO_HYPERPARAMS["hidden_size"],
        )
        trainer = PPOTrainer(
            env=env,
            agent=agent,
            trajectory_length=PPO_HYPERPARAMS["trajectory_length"],
            epochs_per_update=PPO_HYPERPARAMS["epochs_per_update"],
        )

        total_timesteps = 0
        update_count = 0

        while total_timesteps < num_timesteps:
            trajectory = trainer.collect_trajectory(PPO_HYPERPARAMS["trajectory_length"])
            total_timesteps += PPO_HYPERPARAMS["trajectory_length"]

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
                epochs=PPO_HYPERPARAMS["epochs_per_update"],
            )
            update_count += 1

            if log_to_wandb and update_count % 5 == 0:
                log_wandb(
                    log_to_wandb,
                    {
                        "timesteps": total_timesteps,
                        "actor_loss": update_metrics["actor_loss"],
                        "critic_loss": update_metrics["critic_loss"],
                        "entropy": update_metrics["entropy"],
                        "clip_fraction": update_metrics["clip_fraction"],
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
                    agent.save(str(best_checkpoint_path), hyperparams=PPO_HYPERPARAMS)
                print(
                    f"  [update {update_count:4d} | steps {total_timesteps:>10,}] "
                    f"eval={stats['mean_reward']:.2f}  best={best_eval_reward:.2f}"
                )
                if log_to_wandb:
                    log_wandb(
                        log_to_wandb,
                        {
                            "timesteps": total_timesteps,
                            "eval_mean_reward": stats["mean_reward"],
                            "best_eval_reward": best_eval_reward,
                        },
                        step=total_timesteps,
                    )

        agent.save(str(checkpoint_path), hyperparams=PPO_HYPERPARAMS)
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

        finish_wandb(log_to_wandb, summary)

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
