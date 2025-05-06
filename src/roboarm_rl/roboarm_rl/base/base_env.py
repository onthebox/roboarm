import math
import random
import time

import gymnasium as gym
import numpy as np
import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, Quaternion
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboarm_rl.base.listeners import CameraNode, JointStateNode


class RoboarmBaseEnv(gym.Env):
    # metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, render_mode=None):
        rclpy.init()

        super(RoboarmBaseEnv, self).__init__()

        # Инициализация ROS 2
        self.node = Node('roboarm_env')

        self.joint_node = JointStateNode()
        self.camera_node = CameraNode()

        # Запускаем каждую ноду в своем потоке
        self.joint_executor = SingleThreadedExecutor()
        self.joint_executor.add_node(self.joint_node)

        self.camera_executor = SingleThreadedExecutor()
        self.camera_executor.add_node(self.camera_node)

        self.action_space = spaces.Dict({
            "arm_positions": spaces.Box(low=-1.57, high=1.57, shape=(5,), dtype=np.float32),
            "gripper_positions": spaces.Box(low=0.0, high=0.05, shape=(1,), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            "joint_pos": spaces.Box(low=-1.57, high=1.57, shape=(5,), dtype=np.float32),
            "camera_image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        })

        # ROS 2 интерфейсы
        self._init_ros_connections()

        # Для визуализации
        self.render_mode = render_mode
        self.viewer = None
        # Добавляем клиент для спавна объектов
        self.spawn_entity_client = self.node.create_client(
            SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.node.create_client(DeleteEntity, '/delete_entity')
        
        # Параметры куба
        self.cube_sdf_path = '/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/cube.sdf'  # Укажите полный путь
        self.cube_name = 'blue_cube'
        self.current_cube = None

    def _spawn_cube(self):
        """Спавнит куб на окружности радиусом 0.5м"""
        # Удаляем предыдущий куб (если есть) и ждем завершения
        if self.current_cube:
            if not self._delete_entity(self.current_cube):
                self.node.get_logger().warn(f"Failed to delete {self.current_cube}")
            self.current_cube = None
        
        # Генерация случайного угла и позиции
        angle = random.uniform(0, 2 * math.pi)
        radius = 2.5
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        # Подготовка Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.05
        pose.orientation = self._yaw_to_quaternion(angle + math.pi)
        
        # Спавн нового куба
        req = SpawnEntity.Request()
        req.name = self.cube_name
        req.xml = open(self.cube_sdf_path, 'r').read()
        req.initial_pose = pose
        
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        
        if future.result() is not None and future.result().success:
            self.current_cube = self.cube_name
            self.node.get_logger().info(f"Cube spawned at ({x:.2f}, {y:.2f})")
            return True
        else:
            error_msg = future.result().status_message if future.result() else "Timeout"
            self.node.get_logger().error(f"Failed to spawn cube: {error_msg}")
            return False

    def _delete_entity(self, name):
        """Удаляет объект из симуляции с подтверждением"""        
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        
        if future.result() is not None:
            if future.result().success:
                self.node.get_logger().info(f"Deleted entity: {name}")
                return True
            else:
                self.node.get_logger().warn(f"Delete failed: {future.result().status_message}")
        else:
            self.node.get_logger().warn("Delete request timeout")
        return False
    
    def _yaw_to_quaternion(self, yaw):
        """Преобразует угол yaw в Quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2)
        q.w = math.cos(yaw / 2)
        return q

    def _init_ros_connections(self):
        """Инициализация ROS 2 подписчиков"""

        self.arm_action_pub = self.node.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_action_pub = self.node.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.node.get_logger().info("ROS connections initialized")

    def step(self, action):
        # Отправка действия
        self.node.get_logger().info(f"Step action: {action}")
        self._publish_action(action)

        start_time = time.time()
        while time.time() - start_time < 1.0:  # Таймаут 1 сек
            # Проверяем оба топика
            self.joint_executor.spin_once(timeout_sec=0.01)  # Неблокирующий вызов
            self.camera_executor.spin_once(timeout_sec=0.1)

            # Если оба сообщения получены — выходим
            if self.joint_node.has_new_data and self.camera_node.has_new_data:
                break
        # Ожидание обновления
        self._last_joint_state = self.joint_node.get_state()
        self._last_image_state = self.camera_node.get_image()

        # Получение наблюдения
        obs = self._get_obs()

        # Вычисление награды
        reward = self._calculate_reward()

        # Проверка завершения
        terminated = False
        truncated = False

        # TEST SPAWN CUBE 
        self._spawn_cube()

        # Инфо для отладки
        # info = {"is_success": self._check_grasp()}

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

        self.node.get_logger().info(f"Action published: {action}")

    def _get_obs(self):
        """Сбор наблюдений из коллбеков"""
        return {
            "joint_pos": self._last_joint_state,
            "camera_image": self._last_image_state
        }

    def _calculate_reward(self):
        return 1

    def close(self):
        self.joint_node.destroy_node()
        self.camera_node.destroy_node()
        rclpy.shutdown()
