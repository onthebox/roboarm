import time

# import cv2
import gymnasium as gym
import numpy as np
import rclpy
from cv_bridge import CvBridge
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class JointStateNode(Node):
    def __init__(self):
        super().__init__('joint_state_node')
        self._last_joint_state = np.zeros(6, dtype=np.float32)
        self._has_new_data = False
        self._subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            10
        )

    def callback(self, msg):

        try:
            self._last_joint_state = np.array(msg.position, dtype=np.float32)
            self.has_new_data = True
            self.get_logger().info(f"Joint state updated: {self._last_joint_state}")
        except Exception as e:
            self.get_logger().error(f"Joint state error: {e}")

    def get_state(self):
        if not self.has_new_data:
            return None
        self.has_new_data = False
        return self._last_joint_state.copy()

    @property
    def has_new_data(self):
        return self._has_new_data

    @has_new_data.setter
    def has_new_data(self, value):
        assert isinstance(value, bool), "Value must be a boolean"
        self._has_new_data = value


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self._last_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._has_new_data = False
        self.bridge = CvBridge()
        self._subscription = self.create_subscription(
            Image,
            '/roboeye/camera/image_raw',
            self.callback,
            10
        )

    def callback(self, msg):

        self.get_logger().info("Hello from image callback!!!")
        try:
            # Конвертация ROS Image -> OpenCV
            self._last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.has_new_data = True
            self.get_logger().info(f"Image received: {type(self._last_image)}")

            # # Предобработка (изменение размера и обрезка)
            # resized = cv2.resize(cv_image, (84, 84))
            # self.last_image = resized[..., :3]  # На случай альфа-канала

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def get_image(self):

        if not self.has_new_data:
            return None
        self.has_new_data = False
        return self._last_image.copy()

    @property
    def has_new_data(self):
        return self._has_new_data

    @has_new_data.setter
    def has_new_data(self, value):
        assert isinstance(value, bool), "Value must be a boolean"
        self._has_new_data = value


class RoboarmEnv(gym.Env):
    # metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, render_mode=None):
        rclpy.init()

        super(RoboarmEnv, self).__init__()

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

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)

    #     # Сброс симуляции
    #     self._reset_simulation()

    #     # Возврат наблюдения
    #     obs = self._get_obs()
    #     info = {"reset_status": "success"}

    #     return obs, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._get_camera_image()
    #     return None

    # def close(self):
    #     if self.viewer is not None:
    #         self.viewer.close()
    #     rclpy.shutdown()


def main():

    env = RoboarmEnv()

    sample_action = env.action_space.sample()
    env.step(sample_action)
    env.close()

    # rclpy.shutdown()


if __name__ == '__main__':
    main()
