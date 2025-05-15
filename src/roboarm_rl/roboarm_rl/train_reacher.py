import csv
import glob
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from roboarm_rl.reacher_utils.reacher_env import RoboarmReacherEnv


class LoggingCallback(BaseCallback):
    def __init__(self, log_dir: str = "logs", check_freq: int = 1000, save_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.step_rewards = []
        self.episode_lengths = []
        self.current_episode_length = 0
        self.keep_last = 9

        # Создаем файл для записи
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
        self.checkpoint_dir = os.path.join(log_dir, f"checkpoints_{timestamp}")

        # Создаем директорию для логов, если ее нет
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Записываем заголовки в CSV
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode',
                'total_steps',
                'episode_reward',
                'episode_length',
                'mean_reward_100',
                'mean_length_100'
            ])

    def _on_step(self) -> bool:
        self.current_episode_length += 1

        # Записываем награду за шаг
        self.step_rewards.append(self.locals["rewards"][0])

        # Обработка завершения эпизода
        if self.locals['dones'][0]:
            episode_reward = sum(self.step_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_length = 0
            self.step_rewards = []

            # Вычисляем скользящие средние
            mean_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0
            mean_length_100 = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) > 0 else 0

            # Записываем данные в CSV
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    len(self.episode_rewards),
                    self.num_timesteps,
                    episode_reward,
                    self.episode_lengths[-1],
                    mean_reward_100,
                    mean_length_100
                ])

                # Сохранение чекпоинта
        if self.num_timesteps % self.save_freq == 0:
            if len(os.listdir(self.checkpoint_dir)) > self.keep_last:
                oldest = min(
                    glob.glob(os.path.join(self.checkpoint_dir, '*')),
                    key=os.path.getctime
                    )
                os.remove(os.path.join(self.checkpoint_dir, oldest))
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"model_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_path)

        return True


def main():
    # Создаем среду
    env = RoboarmReacherEnv(num_steps=20)

    # Шум для DDPG
    noise_sigma = 0.3 * np.ones(5)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(5),
        sigma=noise_sigma,
        theta=0.1,
        dt=1e-2
    )

    # Инициализация DDPG
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=5000,
        device="auto"
    )

    # logger = CSVOutputFormat("/home/vitya/diploma/roboarm/src/roboarm_rl/training_logs/test.csv")
    # logger = configure("/home/vitya/diploma/roboarm/src/roboarm_rl/training_logs", ["stdout", "csv"])
    # model.set_logger(logger)

    # Callback для логирования
    logging_callback = LoggingCallback(
        log_dir="/home/vitya/diploma/roboarm/src/roboarm_rl/training_logs/20250515",
        save_freq=10
        )

    # Обучение с callback
    model.learn(total_timesteps=50_000, callback=logging_callback)

    # Сохранение модели
    model.save("ddpg_custom_env")


if __name__ == '__main__':
    main()
