import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from roboarm_rl.base import LoggingCallback
from roboarm_rl.reacher_utils import RoboarmReacherVecEnv


def main():
    """Run train."""

    env = RoboarmReacherVecEnv(num_steps=100, observation_type='emb')

    # Action noise
    noise_sigma = 0.03 * np.ones(4)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(4),
        sigma=noise_sigma,
        theta=0.1,
        dt=1e-1
    )

    # Init model
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=5e-4,
        buffer_size=25_000,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512], qf=[512, 512]),
            optimizer_kwargs=dict(weight_decay=1e-5),
        ),
        device="auto",
        learning_starts=1000,
    )

    # Logging callback
    logging_callback = LoggingCallback(
        log_dir="path/to/logs",
        )

    # Train
    model.learn(total_timesteps=500_000, callback=logging_callback)


if __name__ == '__main__':
    main()
