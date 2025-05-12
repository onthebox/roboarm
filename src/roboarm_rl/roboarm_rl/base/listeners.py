from typing import Dict, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat


class JointStateListener(Node):
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
            self._last_joint_state = np.array(msg.position, dtype=np.float32)[:6]
            self.has_new_data = True
            self.get_logger().info("New joint state received.")
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


class CameraListener(Node):
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

        try:
            # Конвертация ROS Image -> OpenCV
            self._last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.has_new_data = True
            self.get_logger().info("New image received.")

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


class LinkPoseListener(Node):
    def __init__(self):
        super().__init__('robot_state_monitor')

        # Параметры
        self.global_frame = 'world'  # Основная система координат
        self.robot_links = [
            'body_link',
            'shoulder_link',
            'forearm_link',
            'wrist_link',         # Части робота для мониторинга
            'palm_link',
            'left_finger_link',
            'right_finger_link'
        ]

        # Инициализация TF2
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Хранилище последних состояний
        self.global_poses: Dict[str, dict] = {}

        # Таймер для обновления координат (20 Гц)
        self.update_timer = self.create_timer(0.05, self.update_transforms)

        self.get_logger().info(f"Monitoring {len(self.robot_links)} robot links")

    def update_transforms(self):
        """Обновление глобальных координат для всех частей робота"""
        for link_name in self.robot_links:
            try:
                # Получаем трансформацию в глобальные координаты
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame,
                    link_name,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1))

                self.process_transform(link_name, transform)

            except TransformException as ex:
                self.get_logger().warn(
                    f"Transform {self.global_frame} -> {link_name} failed: {str(ex)}",
                    throttle_duration_sec=5.0)
                continue

    def process_transform(self, link_name: str, transform: TransformStamped):
        """Обработка и сохранение трансформации"""
        t = transform.transform.translation
        r = transform.transform.rotation

        # Сохраняем данные в словарь
        self.global_poses[link_name] = {
            'position': np.array([t.x, t.y, t.z]),
            'orientation': np.array([r.x, r.y, r.z, r.w]),
            'matrix': self.transform_to_matrix(t, r),
            'stamp': transform.header.stamp
        }

    def transform_to_matrix(self, translation, rotation) -> np.ndarray:
        """Преобразование в матрицу 4x4"""
        t = [translation.x, translation.y, translation.z]
        r = [rotation.x, rotation.y, rotation.z, rotation.w]
        return compose(t, quat2mat(r), np.ones(3))

    def get_pose(self, link_name: str) -> Optional[dict]:
        """Получение последних координат для указанной части робота"""
        return self.global_poses.get(link_name, None)

    def get_all_poses(self) -> Dict[str, dict]:
        """Получение всех сохраненных координат"""
        return self.global_poses
