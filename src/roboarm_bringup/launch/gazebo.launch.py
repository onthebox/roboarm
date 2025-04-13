import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    roboticarm_description_dir = get_package_share_directory('roboarm_description')
    roboticarm_description_share = os.path.join(get_package_prefix('roboarm_description'), 'share')
    gazebo_ros_dir = get_package_share_directory('ros_gz_sim')

    model_arg = DeclareLaunchArgument(name='model', default_value=os.path.join(
                                        roboticarm_description_dir, 'urdf', 'roboarm.urdf.xacro'
                                        ),
                                      description='Absolute path to robot urdf file'
    )

    env_var = SetEnvironmentVariable('GAZEBO_MODEL_PATH', roboticarm_description_share)

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gz_sim.launch.py')
        ),
        # launch_arguments={'gz_args': '-r -s -v4 /home/vitya/zero_gravity.world', 'on_exit_shutdown': 'true'}.items()
        launch_arguments={'gz_args': '-r -s -v4 empty.sdf', 'on_exit_shutdown': 'true'}.items()
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v4 '}.items()
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name',
            'roboarm',
            '-topic',
            'robot_description',
            ],
        output='screen'
    )

    return LaunchDescription([
        env_var,
        model_arg,
        start_gazebo_server,
        start_gazebo_client,
        robot_state_publisher_node,
        spawn_robot
    ])