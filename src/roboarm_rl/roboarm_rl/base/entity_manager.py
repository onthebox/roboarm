import math
import random
from typing import List

import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node


class EntityManager(Node):
    def __init__(self, entity_file, entity_name, executor, timeout=30.0):
        super().__init__('entity_manager_node')
        self.entity_file = entity_file
        self.entity_name = entity_name
        self._spawn_entity = self.create_client(SpawnEntity, "/spawn_entity")
        self._delete_entity = self.create_client(DeleteEntity, "/delete_entity")
        self._executor = executor
        self._timeout = timeout
        self._current_entity = None

    def spawn(self, randomize: bool = False, position: List[float] | None = None):

        assert randomize ^ bool(position), "Only one of 'randomize' or 'position' can be specified"

        # Удаляем предыдущий куб (если есть) и ждем завершения
        if self._current_entity:
            if not self.delete():
                self.get_logger().warn(f"Failed to delete {self._current_entity}")
            self._current_entity = None

        if randomize:
            # sector_width = math.radians(90)
            # Генерация случайного угла и позиции
            angle = math.radians(random.uniform(-90, 0))
            radius = 1.5
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.05

        # Подготовка Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        # face_angle = math.atan2(-y, -x)  # Направление к центру (0,0)
        # pose.orientation = self._yaw_to_quaternion(face_angle)

        # Спавн нового куба
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = open(self.entity_file, 'r').read()
        req.initial_pose = pose

        future = self._spawn_entity.call_async(req)
        rclpy.spin_until_future_complete(self, future, executor=self._executor, timeout_sec=self._timeout)

        if future.result() is not None and future.result().success:
            self._current_entity = self.entity_name
            self.get_logger().info(f"Cube spawned at ({x:.2f}, {y:.2f}, {z:.2f})")
            return [x, y, z]
        else:
            error_msg = future.result().status_message if future.result() else "Timeout"
            self.get_logger().error(f"Failed to spawn cube: {error_msg}")
            return False

    def _yaw_to_quaternion(self, yaw):
        """Преобразует угол yaw в Quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2)
        q.w = math.cos(yaw / 2)
        return q

    def delete(self):
        """Удаляет объект из симуляции с подтверждением"""
        req = DeleteEntity.Request()
        req.name = self.entity_name
        future = self._delete_entity.call_async(req)
        rclpy.spin_until_future_complete(self, future, executor=self.executor, timeout_sec=self._timeout)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f"Deleted entity: {self.entity_name}")
                return True
            else:
                self.get_logger().warn(f"Delete failed: {future.result().status_message}")
        else:
            self.get_logger().warn("Delete request timeout")
        return False
