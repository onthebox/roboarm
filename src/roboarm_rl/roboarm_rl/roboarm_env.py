import time

# from roboarm_rl.base import RoboarmBaseEnv
from roboarm_rl.reacher_utils.reacher_env import RoboarmReacherEnv

# from stable_baselines3.common.env_checker import check_env


def main():

    env = RoboarmReacherEnv()
    # check_env(env)
    env.reset()
    sample_action = env.action_space.sample()
    env.step(sample_action)
    time.sleep(5.0)
    env.reset()
    env.close()


if __name__ == '__main__':
    main()
