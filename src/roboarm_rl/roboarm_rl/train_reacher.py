import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from roboarm_rl.reacher_utils.reacher_env import RoboarmReacherEnv


def main():

    # Создаем среду
    env = RoboarmReacherEnv()
    # check_env(env)  # Проверка совместимости

    # Шум для DDPG (улучшает исследование)
    noise_sigma = 0.2 * np.ones(5)  # Подберите под вашу задачу
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(5),
        sigma=noise_sigma,
        theta=0.15,
        dt=1e-2
    )

    # Инициализация DDPG
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        buffer_size=100_000,  # Размер replay buffer
        batch_size=64,        # Размер батча для обучения
        device="auto"         # Автовыбор CPU/GPU
    )

    # Обучение (50,000 шагов)
    model.learn(total_timesteps=50_000)

    # Сохранение модели
    model.save("ddpg_custom_env")

    # # Загрузка и тестирование
    # del model  # Удаляем для чистоты теста
    # model = DDPG.load("ddpg_custom_env")

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()


if __name__ == '__main__':
    main()
