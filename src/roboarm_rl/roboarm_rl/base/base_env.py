import time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboarm_rl.base.entity_manager import EntityManager
from roboarm_rl.base.listeners import (
    CameraListener,
    JointStateListener,
    LinkPoseListener,
)


class RoboarmBaseEnv(gym.Env):
    def __init__(self, num_steps: int = 200):
        rclpy.init()

        super(RoboarmBaseEnv, self).__init__()

        # Инициализация ноды среды
        self.node = Node('roboarm_env')

        # Создание нод для получения состояний робота
        self.joint_listener = JointStateListener()
        self.camera_listener = CameraListener()
        self.link_pose_listener = LinkPoseListener()

        # Создание ноды для управления объектами
        self.entity_manager_executor = SingleThreadedExecutor()
        self.entity_manager = EntityManager(
            entity_file='/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/cube.sdf',
            entity_name='blue_cube',
            executor=self.entity_manager_executor
        )
        self.entity_manager_executor.add_node(self.entity_manager)

        # Запускаем каждую ноду в своем потоке
        self.joint_executor = SingleThreadedExecutor()
        self.joint_executor.add_node(self.joint_listener)

        self.camera_executor = SingleThreadedExecutor()
        self.camera_executor.add_node(self.camera_listener)

        self.link_pose_executor = SingleThreadedExecutor()
        self.link_pose_executor.add_node(self.link_pose_listener)

        self.action_space = spaces.Dict({
            "arm_positions": spaces.Box(low=-1.57, high=1.57, shape=(5,), dtype=np.float32),
            "gripper_positions": spaces.Box(low=0.0, high=0.05, shape=(1,), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            "joint_pos": spaces.Box(low=-1.57, high=1.57, shape=(6,), dtype=np.float32),
            "camera_image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        })

        self._init_publishers()

        self.num_steps = num_steps
        self._current_step = 0

    def _init_publishers(self):
        """Инициализация ROS 2 подписчиков"""

        self.arm_action_pub = self.node.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_action_pub = self.node.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.node.get_logger().info("ROS connections initialized")

    def step(self, action):
        # Отправка действия
        self.node.get_logger().info("Step start.")
        self.node.get_logger().info(f"Action:\n{action}")
        self._publish_action(action)

        # Получение наблюдения
        obs = self._get_obs()

        self._current_step += 1

        # Вычисление награды
        reward = self._calculate_reward()

        # Проверка завершения
        terminated = False
        truncated = self._current_step > self.num_steps

        self.node.get_logger().info(f"Step end. Terminated: {terminated}. Truncated: {truncated}.")

        return obs, reward, terminated, truncated, {}

    def _publish_action(self, action):
        """Отправка действия в симуляцию"""
        # Создание сообщений для публикации
        arm_msg = JointTrajectory()
        gripper_msg = JointTrajectory()

        arm_msg.joint_names = [
            'base_body_joint', 'body_root1_joint',
            'shoulder_root2_joint', 'forearm_root3_joint', 'wrist_root4_joint'
            ]
        gripper_msg.joint_names = ['palm_left_finger_joint']

        arm_traj_point = JointTrajectoryPoint()
        gripper_traj_point = JointTrajectoryPoint()

        arm_traj_point.positions = action["arm_positions"].tolist()
        gripper_traj_point.positions = action["gripper_positions"].tolist()

        arm_msg.points.append(arm_traj_point)
        gripper_msg.points.append(gripper_traj_point)

        self.arm_action_pub.publish(arm_msg)
        self.gripper_action_pub.publish(gripper_msg)

    def _get_obs(self):
        """Сбор наблюдений из коллбеков"""

        start_time = time.time()
        while time.time() - start_time < 1.0:  # Таймаут 1 сек
            # Проверяем оба топика
            self.joint_executor.spin_once(timeout_sec=0.01)  # Неблокирующий вызов
            self.camera_executor.spin_once(timeout_sec=0.1)

            # Если оба сообщения получены — выходим
            if self.joint_listener.has_new_data and self.camera_listener.has_new_data:
                break

        self._last_joint_state = self.joint_listener.get_state()
        self._last_image_state = self.camera_listener.get_image()

        return {
            "joint_pos": self._last_joint_state,
            "camera_image": self._last_image_state
        }

    def _calculate_reward(self):
        start_time = time.time()
        while time.time() - start_time < 1.0:  # Таймаут 1 сек
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            palm_pos = self.link_pose_listener.get_pose('palm_link')

        self.node.get_logger().info(f'Palm position: {palm_pos}')
        return 1

    def reset(self, seed=0):
        self._current_step = 0
        init_pose = {
            'arm_positions': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'gripper_positions': np.array([0.0], dtype=np.float32)
        }
        self._publish_action(init_pose)
        self.entity_manager.delete()
        time.sleep(5)
        self._entity_position = self.entity_manager.spawn(randomize=True)
        return self._get_obs(), {}

    def render(self):
        # Возвращаем последнюю картинку
        return self._last_image_state

    def close(self):
        self.joint_listener.destroy_node()
        self.camera_listener.destroy_node()
        self.link_pose_listener.destroy_node()
        self.entity_manager.destroy_node()
        rclpy.shutdown()
