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
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    model_arg = DeclareLaunchArgument(
        name='model',
        default_value=os.path.join(
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
            os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': '/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/custom_empty.world'  # Явный путь
        }.items()
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')
        )
    )

    spawn_ground = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'ground_plane',
            '-file', '/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/ground.sdf',
            '-z', '0.0'  # Высота (Z=0)
        ],
        output='screen'
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',
            'roboarm',
            '-topic',
            'robot_description',
            ],
        output='screen'
        )

    spawn_cube = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',
            'blue_cube',
            '-file',
            '/home/vitya/diploma/roboarm/src/roboarm_bringup/entities/cube.sdf',
            '-x', '1.0606601717798214',
            '-y', '1.0606601717798214',
            '-z', '0.05',
        ],
        output='screen'
    )

    return LaunchDescription([
        env_var,
        model_arg,
        start_gazebo_server,
        start_gazebo_client,
        robot_state_publisher_node,
        spawn_ground,
        spawn_robot,
        spawn_cube
    ])
