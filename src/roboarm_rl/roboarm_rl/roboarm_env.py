import time

# from roboarm_rl.base import RoboarmBaseEnv
from roboarm_rl.reacher_utils.reacher_env import RoboarmReacherEnv


def main():

    env = RoboarmReacherEnv()

    env.reset()
    sample_action = env.action_space.sample()
    env.step(sample_action)
    time.sleep(1.0)
    env.close()


if __name__ == '__main__':
    main()
