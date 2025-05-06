from roboarm_rl.base import RoboarmBaseEnv


def main():

    env = RoboarmBaseEnv()

    sample_action = env.action_space.sample()
    env.step(sample_action)
    env.close()


if __name__ == '__main__':
    main()
