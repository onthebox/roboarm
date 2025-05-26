import csv
import os
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    """Callback class for saving logs and checlpoints while training."""
    def __init__(self, log_dir: str = "logs", verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.step_rewards = []
        self.episode_lengths = []
        self.current_episode_length = 0
        self.best_reward = -np.inf

        # Make log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
        self.checkpoint_dir = os.path.join(log_dir, f"checkpoints_{timestamp}")

        # Make directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Write csv header
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode',
                'total_steps',
                'episode_reward',
                'episode_length',
                'mean_reward_100',
                'mean_length_100',
                "actor_loss",
                "critic_loss",
                "learning_rate"
            ])

    def _on_step(self) -> bool:
        """On step logic for the callback."""
        self.current_episode_length += 1

        # Reward for current step
        self.step_rewards.append(self.locals["rewards"][0])

        # Process episode end
        if self.locals['dones'][0]:
            episode_reward = sum(self.step_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_length = 0
            self.step_rewards = []

            # Calculate moving averages
            mean_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0
            mean_length_100 = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) > 0 else 0

            # Write down logs
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    len(self.episode_rewards),
                    self.num_timesteps,
                    episode_reward,
                    self.episode_lengths[-1],
                    mean_reward_100,
                    mean_length_100,
                    self.logger.name_to_value["train/actor_loss"],
                    self.logger.name_to_value["train/critic_loss"],
                    self.logger.name_to_value["train/learning_rate"],
                ])

            # Save weights checkpoints
            last_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                "last.zip"
            )
            self.model.save(last_checkpoint_path)
            if episode_reward > self.best_reward:
                best_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    "best.zip"
                )
                self.model.save(best_checkpoint_path)

        return True
