#!/usr/bin/env python3
"""
complete.launch.py
==================
Launches:
  1. Gazebo Harmonic with the drone_arena world
  2. The X3 drone URDF into Gazebo
  3. robot_state_publisher
  4. ROS-GZ bridge (cmd_vel, odom, camera)
  5. opencv_navigation node
  6. moving_obstacle node
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PKG = 'opencv_drone_vision'


def generate_launch_description():

    pkg_share   = get_package_share_directory(PKG)
    world_file  = os.path.join(pkg_share, 'worlds', 'drone_arena.sdf')
    urdf_file   = os.path.join(pkg_share, 'urdf',   'x3_drone.urdf')

    # ── Arguments ─────────────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock')

    gz_verbose_arg = DeclareLaunchArgument(
        'gz_verbose', default_value='false',
        description='Gazebo verbose logging')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── 1. Gazebo Harmonic ─────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            world_file,
        ],
        output='screen',
        additional_env={'GZ_SIM_SYSTEM_PLUGIN_PATH': '/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins'}
    )

    # ── 2. Spawn drone ──────────────────────────────────────────────────────────
    spawn_drone = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz', 'service', '-s', '/world/drone_arena/create',
                    '--reqtype', 'gz.msgs.EntityFactory',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '5000',
                    '--req',
                    (
                        'sdf_filename: "' + urdf_file + '" '
                        'name: "x3" '
                        'pose { position { x: 0 y: 0 z: 0.2 } '
                        '       orientation { w: 1 } }'
                    ),
                ],
                output='screen'
            )
        ]
    )

    # ── 3. robot_state_publisher ───────────────────────────────────────────────
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # ── 4. ROS-GZ bridge ───────────────────────────────────────────────────────
    # Maps Gazebo topics → ROS 2 topics
    gz_ros_bridge = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2',
                    'run',
                    'ros_gz_bridge',
                    'parameter_bridge',

                    '/x3/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',

                    '/model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',

                    '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',

                    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

                    '/x3/enable@std_msgs/msg/Bool]gz.msgs.Boolean'
                ],
                output='screen'
            )
        ]
    )

    # ── 5. OpenCV Navigation node ─────────────────────────────────────────────
    navigation_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG,
                executable='opencv_navigation',
                name='opencv_navigation',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[
                    ('/drone/camera/image_raw', '/drone/camera/image_raw'),
                    ('/drone/cmd_vel',          '/drone/cmd_vel'),
                    ('/drone/odom',             '/drone/odom'),
                ]
            )
        ]
    )

    # ── 6. Moving obstacle node ───────────────────────────────────────────────
    obstacle_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG,
                executable='moving_obstacle',
                name='moving_obstacle',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        gz_verbose_arg,
        gazebo,
        rsp_node,
        spawn_drone,
        gz_ros_bridge,
        navigation_node,
        obstacle_node,
    ])
