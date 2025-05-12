import time

import numpy as np

from roboarm_rl.base import RoboarmBaseEnv


class RoboarmReacherEnv(RoboarmBaseEnv):

    def _calculate_reward(self):

        # Get roboarm palm current position
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            palm_pos = self.link_pose_listener.get_pose('palm_link')

        self.node.get_logger().info('Calculating reward...')
        # Distance from palm of the robot to the target object
        distance = np.linalg.norm(np.array(self._entity_position) - palm_pos['position'])
        # Distance from object to the origin
        max_distance = np.linalg.norm(np.array(self._entity_position))
        normalized_distance = distance / max_distance

        distance_reward = (1 - normalized_distance) * self.num_steps
        reward = distance_reward - self._current_step

        self.node.get_logger().info(f'Distance to the object: {distance}')
        self.node.get_logger().info(f'Reward from distance: {distance_reward}')
        self.node.get_logger().info(f'Overall reward: {reward}')

        return reward
