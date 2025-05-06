import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


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
