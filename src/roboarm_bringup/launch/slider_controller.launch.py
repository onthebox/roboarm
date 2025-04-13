import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    controller = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("roboarm_bringup"),
                "launch",
                "controller.launch.py"
            ),
            launch_arguments={"is_sim": "True"}.items()
        )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        remappings=[
            ('/joint_states', '/joint_commands'),
        ]
    )

    slider_control_node = Node(
        package='roboarm_controller',
        executable='slider_controller.py'
    )

    return LaunchDescription(
        [
            controller,
            joint_state_publisher_gui_node,
            slider_control_node
        ]
    )
