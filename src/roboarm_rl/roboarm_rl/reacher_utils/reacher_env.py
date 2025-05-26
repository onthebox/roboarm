import time
from typing import Tuple

import cv2
import numpy as np
import rclpy
import torch
import torchvision.models as models
from gymnasium import spaces
from PIL import Image
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from torchvision import transforms
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboarm_rl.base import RoboarmBaseEnv
from roboarm_rl.base.entity_manager import EntityManager
from roboarm_rl.base.listeners import (
    CameraListener,
    JointStateListener,
    LinkPoseListener,
)


class RoboarmReacherVecEnv(RoboarmBaseEnv):
    """Gym environment for the Roboarm Reacher task.

    The observation space is a vector of shape (18,), consisting of:
    - 4 joint angles cos
    - 4 joint angles sin
    - 4 joint velocities (rad/step)
    - 3 coords of a vector between palm and a target (x, y, z)
    - 3 target coords (x, y, z)
    """

    def __init__(self, num_steps: int = 200) -> None:
        rclpy.init()

        super(RoboarmBaseEnv, self).__init__()
        # Init self node
        self.node = Node('roboarm_env')

        # Init listeners node
        self.joint_listener = JointStateListener()
        self.link_pose_listener = LinkPoseListener()

        # Init entity manager node
        self.entity_manager_executor = SingleThreadedExecutor()
        self.entity_manager = EntityManager(
            entity_file='/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/cube.sdf',
            entity_name='blue_cube',
            executor=self.entity_manager_executor
        )
        self.entity_manager_executor.add_node(self.entity_manager)

        # Launch nodes as separate threads
        self.joint_executor = SingleThreadedExecutor()
        self.joint_executor.add_node(self.joint_listener)

        self.link_pose_executor = SingleThreadedExecutor()
        self.link_pose_executor.add_node(self.link_pose_listener)

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # Init publisher
        self._init_publishers()

        self.num_steps = num_steps
        self._current_step = 0
        self._action_trace = np.zeros((4, 4))

    def _get_obs(self) -> np.array:
        """Get observation vector.

        Returns:
            np.array: observation
        """
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.joint_executor.spin_once(timeout_sec=0.01)

            if self.joint_listener.has_new_data:
                break

        self._last_joint_state = self.joint_listener.get_state()

        angles = []

        for c, s in zip(np.cos(self._last_joint_state[:4]), np.sin(self._last_joint_state[:4])):
            angles.append(c)
            angles.append(s)

        velocities = np.sum(self._action_trace, axis=0)
        vec_to_cube = np.array(self._entity_position) - self._last_palm_pos['position']

        obs = np.concatenate(
            [
                np.array(angles),
                velocities,
                vec_to_cube,
                np.array(self._entity_position)
            ]
        )

        return obs

    def _calculate_reward(self) -> float:
        """Calculate agent reward.

        Returns:
            float: reward
        """
        # Get roboarm palm current position
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            self._last_palm_pos = self.link_pose_listener.get_pose('palm_link')

        self.node.get_logger().info('Calculating reward...')
        # Distance from palm of the robot to the target object
        self._last_distance = np.linalg.norm(np.array(self._entity_position) - self._last_palm_pos['position'])

        overpushed_reward = -np.linalg.norm(np.abs(self._last_action)) / \
            np.linalg.norm(np.abs(np.array([0.1, 0.1, 0.1, 0.1])))

        distance_reward = -self._last_distance
        reward = distance_reward + 0.1 * overpushed_reward

        self.node.get_logger().info(f'Reward from distance: {distance_reward}')
        self.node.get_logger().info(f'Overpushed reward: {overpushed_reward}')
        self.node.get_logger().info(f'Overall reward: {reward}')

        return reward

    def _publish_action(self, action, reset=False) -> None:
        """Publish agent action.

        Args:
            action (np.array): action to publish
            reset (bool): whether to reset the robot, defaults to False
        """
        arm_msg = JointTrajectory()
        self.node.get_logger().info(f"Got action: {action}")
        arm_msg.joint_names = [
            'base_body_joint', 'body_root1_joint',
            'shoulder_root2_joint', 'forearm_root3_joint', 'wrist_root4_joint'
            ]

        arm_traj_point = JointTrajectoryPoint()

        if not reset:
            self._desired_position = self._last_joint_state[:4] + action
            target_position = np.clip(
                self._desired_position,
                a_min=np.array([-1.57, 0.0, 0.0, -1.57]),
                a_max=np.array([0.0, 1.57, 1.57, 1.57])
                )
        else:
            target_position = action

        arm_traj_point.positions = target_position.tolist() + [1.571]

        arm_msg.points.append(arm_traj_point)

        self.arm_action_pub.publish(arm_msg)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """Make a step in the environment.
        Args:
            action (np.array): action to publish

        Returns:
            Tuple[np.array, float, bool, bool, dict]: tuple of (observation, reward, terminated, truncated, info)
        """
        # Publish action
        self.node.get_logger().info(f"Step {self._current_step} start.")
        self._last_action = action
        self._action_trace = np.concatenate([self._action_trace, self._last_action[None, :]])[1:, :]
        self._publish_action(self._last_action)
        time.sleep(0.1)
        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        terminated = self._last_distance <= 0.65
        truncated = self._current_step == self.num_steps - 1
        info = {}

        self.node.get_logger().info(f"Step {self._current_step} end. Terminated: {terminated}. Truncated: {truncated}.")

        self._current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = 0) -> Tuple[np.array, dict]:
        """Reset the environment

        Args:
            seed (int, optional): Gym needs this for some reason. Defaults to 0.

        Returns:
            Tuple[np.array, dict]: Observation and info dict
        """
        self._current_step = 0
        init_pose = np.array([0.0, 1.225, 1.57, 1.57], dtype=np.float32)
        self._publish_action(init_pose, reset=True)
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            self._last_palm_pos = self.link_pose_listener.get_pose('palm_link')
        self.entity_manager.delete()
        time.sleep(5)
        self._entity_position = self.entity_manager.spawn(randomize=True)

        return self._get_obs(), {}


class RoboarmReacherImgEnv(RoboarmBaseEnv):
    """Gym environment for the Roboarm Reacher task.

    The observation space is an image embedding vector.
    """

    def __init__(self, num_steps: int = 200) -> None:
        rclpy.init()

        super(RoboarmBaseEnv, self).__init__()
        # Init self node
        self.node = Node('roboarm_env')

        # Init listeners node
        self.joint_listener = JointStateListener()
        self.camera_listener = CameraListener()
        self.link_pose_listener = LinkPoseListener()

        # Init entity manager node
        self.entity_manager_executor = SingleThreadedExecutor()
        self.entity_manager = EntityManager(
            entity_file='/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/cube.sdf',
            entity_name='blue_cube',
            executor=self.entity_manager_executor
        )
        self.entity_manager_executor.add_node(self.entity_manager)

        # Launch nodes as separate threads
        self.joint_executor = SingleThreadedExecutor()
        self.joint_executor.add_node(self.joint_listener)

        self.camera_executor = SingleThreadedExecutor()
        self.camera_executor.add_node(self.camera_listener)

        self.link_pose_executor = SingleThreadedExecutor()
        self.link_pose_executor.add_node(self.link_pose_listener)

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(576,), dtype=np.float32)

        # Init publisher and image processor
        self._init_image_processor()
        self._init_publishers()

        self.num_steps = num_steps
        self._current_step = 0

    def _init_image_processor(self) -> None:
        """Load image processor model (MobileNetV3)."""
        self._model = models.mobilenet_v3_small(pretrained=True)
        self._model.eval()
        self._model.classifier = torch.nn.Identity()

    def _get_embedding(self, image: np.array) -> np.array:
        """Process image to get embedding vector.

        Args:
            - image (np.array): image to process

        Returns:
            - embedding (np.array): embedding vector
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(Image.fromarray(image)).unsqueeze(0).float()
        with torch.no_grad():
            embedding = self._model(image).squeeze(0).numpy()

        normalized_embedding = self._normalize_embedding(embedding)
        return normalized_embedding

    def _normalize_embedding(self, embedding: np.array) -> np.array:
        """Normalize embedding vector.

        Args:
            embedding (np.array): image embedding vector

        Returns:
            np.array: normalized embedding vector
        """
        norm = np.linalg.norm(embedding)

        return embedding / (norm + 1e8)

    def _get_obs(self) -> np.array:
        """Get image embedding as observation.

        Returns:
            np.array: observation
        """
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.joint_executor.spin_once(timeout_sec=0.01)
            self.camera_executor.spin_once(timeout_sec=0.1)

            if self.joint_listener.has_new_data and self.camera_listener.has_new_data:
                break

        self._last_joint_state = self.joint_listener.get_state()
        last_image = self.camera_listener.get_image()
        last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

        embedding = self._get_embedding(last_image)
        self._last_image_state = embedding

        return self._last_image_state

    def _calculate_reward(self) -> float:
        """Calculate agent reward.

        Returns:
            float: reward
        """
        # Get roboarm palm current position
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            self._last_palm_pos = self.link_pose_listener.get_pose('palm_link')

        self.node.get_logger().info('Calculating reward...')
        # Distance from palm of the robot to the target object
        self._last_distance = np.linalg.norm(np.array(self._entity_position) - self._last_palm_pos['position'])

        overpushed_reward = -np.linalg.norm(np.abs(self._last_action)) / \
            np.linalg.norm(np.abs(np.array([0.1, 0.1, 0.1, 0.1])))

        distance_reward = -self._last_distance
        reward = distance_reward + 0.1 * overpushed_reward

        self.node.get_logger().info(f'Reward from distance: {distance_reward}')
        self.node.get_logger().info(f'Overpushed reward: {overpushed_reward}')
        self.node.get_logger().info(f'Overall reward: {reward}')

        return reward

    def _publish_action(self, action, reset=False) -> None:
        """Publish agent action.

        Args:
            action (np.array): action to publish
            reset (bool): whether to reset the robot, defaults to False
        """
        arm_msg = JointTrajectory()
        self.node.get_logger().info(f"Got action: {action}")
        arm_msg.joint_names = [
            'base_body_joint', 'body_root1_joint',
            'shoulder_root2_joint', 'forearm_root3_joint', 'wrist_root4_joint'
            ]

        arm_traj_point = JointTrajectoryPoint()

        if not reset:
            self._desired_position = self._last_joint_state[:4] + action
            target_position = np.clip(
                self._desired_position,
                a_min=np.array([-1.57, 0.0, 0.0, -1.57]),
                a_max=np.array([0.0, 1.57, 1.57, 1.57])
                )
        else:
            target_position = action

        arm_traj_point.positions = target_position.tolist() + [1.571]

        arm_msg.points.append(arm_traj_point)

        self.arm_action_pub.publish(arm_msg)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """Make a step in the environment.
        Args:
            action (np.array): action to publish

        Returns:
            Tuple[np.array, float, bool, bool, dict]: tuple of (observation, reward, terminated, truncated, info)
        """
        # Publish action
        self.node.get_logger().info(f"Step {self._current_step} start.")
        self._last_action = action
        self._publish_action(self._last_action)
        time.sleep(0.1)
        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        terminated = self._last_distance <= 0.65
        truncated = self._current_step == self.num_steps - 1
        info = {}

        self.node.get_logger().info(f"Step {self._current_step} end. Terminated: {terminated}. Truncated: {truncated}.")

        self._current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = 0) -> Tuple[np.array, dict]:
        """Reset the environment

        Args:
            seed (int, optional): Gym needs this for some reason. Defaults to 0.

        Returns:
            Tuple[np.array, dict]: Observation and info dict
        """
        self._current_step = 0
        init_pose = np.array([0.0, 1.225, 1.57, 1.57], dtype=np.float32)
        self._publish_action(init_pose, reset=True)
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.link_pose_executor.spin_once(timeout_sec=0.01)
            self._last_palm_pos = self.link_pose_listener.get_pose('palm_link')
        self.entity_manager.delete()
        time.sleep(5)
        self._entity_position = self.entity_manager.spawn(randomize=True)

        return self._get_obs(), {}
