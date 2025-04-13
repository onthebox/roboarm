#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SliderController(Node):
    def __init__(self):
        super().__init__('slider_controller')

        self.sub_ = self.create_subscription(JointState, '/joint_commands', self.sub_callback, 10)
        self.arm_pub_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub_ = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.get_logger().info('Started Slider Controller')

    def sub_callback(self, msg: JointState):
        arm_msg = JointTrajectory()
        gripper_msg = JointTrajectory()

        arm_msg.joint_names = [
            'base_body_joint', 'body_root1_joint',
            'shoulder_root2_joint', 'forearm_root3_joint', 'wrist_root4_joint'
            ]
        gripper_msg.joint_names = ['palm_left_finger_joint']

        arm_traj_point = JointTrajectoryPoint()
        gripper_traj_point = JointTrajectoryPoint()

        arm_traj_point.positions = [msg.position[i] for i in range(5)]
        gripper_traj_point.positions = [msg.position[5]]

        arm_msg.points.append(arm_traj_point)
        gripper_msg.points.append(gripper_traj_point)

        self.arm_pub_.publish(arm_msg)
        self.gripper_pub_.publish(gripper_msg)


def main():
    rclpy.init()
    slider_controller = SliderController()
    rclpy.spin(slider_controller)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
