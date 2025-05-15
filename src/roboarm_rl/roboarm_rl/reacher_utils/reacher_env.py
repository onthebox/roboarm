import time

import numpy as np
import rclpy
import torch
import torchvision.models as models
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboarm_rl.base import RoboarmBaseEnv
from roboarm_rl.base.entity_manager import EntityManager
from roboarm_rl.base.listeners import (
    CameraListener,
    JointStateListener,
    LinkPoseListener,
)


class RoboarmReacherEnv(RoboarmBaseEnv):

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

        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(5,), dtype=np.float32)

        # self.observation_space = spaces.Dict({
        #     "joint_pos": spaces.Box(low=-1.57, high=1.57, shape=(6,), dtype=np.float32),
        #     "camera_image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        # })
        self.observation_space = spaces.Box(low=0, high=1, shape=(576,), dtype=np.float32)

        self._init_publishers()
        self._init_image_processor()

        self.num_steps = num_steps
        self._current_step = 0

    def _init_image_processor(self):
        self._model = models.mobilenet_v3_small(pretrained=True)
        self._model.eval()
        self._model.classifier = torch.nn.Identity()

    def _get_embedding(self, image):
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            embedding = self._model(image).squeeze(0).numpy()

        normalized_embedding = self._normalize_embedding(embedding)
        return normalized_embedding

    def _normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)

        return embedding / (norm + 1e8)

    def _get_obs(self):
        """Сбор наблюдений из коллбеков"""

        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.camera_executor.spin_once(timeout_sec=0.1)

            # Если оба сообщения получены — выходим
            if self.camera_listener.has_new_data:
                break

        last_image = self.camera_listener.get_image()
        embedding = self._get_embedding(last_image)
        self._last_image_state = embedding

        return self._last_image_state

    def _calculate_reward(self):
        # Get roboarm palm current position
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            self._last_palm_pos = self.link_pose_listener.get_pose('palm_link')

        self.node.get_logger().info('Calculating reward...')
        # Distance from palm of the robot to the target object
        self._last_distance = np.linalg.norm(np.array(self._entity_position) - self._last_palm_pos['position'])
        # Distance from object to the origin
        max_distance = np.linalg.norm(np.array(self._entity_position))
        normalized_distance = self._last_distance / max_distance

        distance_reward = (1 - normalized_distance) * self.num_steps
        reward = distance_reward - self._current_step
        self._cumulative_reward += reward

        self.node.get_logger().info(f'Distance to the object: {self._last_distance}')
        self.node.get_logger().info(f'Reward from distance: {distance_reward}')
        self.node.get_logger().info(f'Overall reward: {reward}')

        return reward

    def _publish_action(self, action):
        """Отправка действия в симуляцию"""
        # Создание сообщений для публикации
        arm_msg = JointTrajectory()
        self.node.get_logger().info(f"Got action: {action}")
        arm_msg.joint_names = [
            'base_body_joint', 'body_root1_joint',
            'shoulder_root2_joint', 'forearm_root3_joint', 'wrist_root4_joint'
            ]

        arm_traj_point = JointTrajectoryPoint()

        arm_traj_point.positions = action.tolist()

        arm_msg.points.append(arm_traj_point)

        self.arm_action_pub.publish(arm_msg)

    def step(self, action):
        # Отправка действия
        self.node.get_logger().info(f"Step {self._current_step} start.")
        self.node.get_logger().info(f"Action:\n{action}")
        self._publish_action(action)

        # Получение наблюдения
        obs = self._get_obs()

        # Вычисление награды
        reward = self._calculate_reward()

        # Проверка завершения
        terminated = self._last_distance <= 0.62
        truncated = self._current_step == self.num_steps - 1
        info = {}

        if terminated or truncated:
            info["episode"] = {
                "r": self._cumulative_reward,  # Суммарная награда за эпизод
                "l": self._current_step     # Длина эпизода в шагах
            }

        self.node.get_logger().info(f"Step {self._current_step} end. Terminated: {terminated}. Truncated: {truncated}.")

        self._current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=0):
        self._cumulative_reward = 0
        self._current_step = 0
        init_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._publish_action(init_pose)
        self.entity_manager.delete()
        time.sleep(5)
        self._entity_position = self.entity_manager.spawn(randomize=True)
        return self._get_obs(), {}
