#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "opencv_drone_vision"


def generate_launch_description():

    pkg_share = get_package_share_directory(PKG)

    world_file = os.path.join(pkg_share, "worlds", "drone_arena.sdf")
    urdf_file = os.path.join(pkg_share, "urdf", "x3_drone.urdf")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    ############################################################
    # Gazebo
    ############################################################

    gazebo = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            world_file,
        ],
        output="screen",
    )

    ############################################################
    # Spawn Drone
    ############################################################

    spawn_drone = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "gz",
                    "service",
                    "-s",
                    "/world/drone_arena/create",
                    "--reqtype",
                    "gz.msgs.EntityFactory",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    "5000",
                    "--req",
                    (
                        f'sdf_filename: "{urdf_file}" '
                        'name: "x3" '
                        'pose { position { x:0 y:0 z:0.2 } orientation { w:1 } }'
                    ),
                ],
                output="screen",
            )
        ],
    )

    ############################################################
    # Robot State Publisher
    ############################################################

    with open(urdf_file, "r") as f:
        robot_description = f.read()

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    ############################################################
    # ROS <-> Gazebo Bridge
    ############################################################

    bridge = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    "/model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    "/x3/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                    "/x3/enable@std_msgs/msg/Bool]gz.msgs.Boolean",
                    "/camera/front/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/camera/down/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
                ],
                output="screen",
            )
        ],
    )

    ############################################################
    # Navigation
    ############################################################

    navigation = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG,
                executable="opencv_navigation_v2",
                name="opencv_navigation_v2",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                    }
                ],
                remappings=[
                    ('/drone/odom',                  '/model/x3/odometry'),
                    ('/drone/cmd_vel',               '/x3/cmd_vel'),
                    ('/drone/enable',                '/x3/enable'),
                    ('/drone/front_camera/image_raw', '/camera/front/image_raw'),
                    ('/drone/down_camera/image_raw',  '/camera/down/image_raw'),
                ],
                output="screen",
            )
        ],
    )

    ############################################################
    # Moving Obstacles
    ############################################################

    moving = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG,
                executable="moving_obstacle",
                name="moving_obstacle",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                    }
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        gazebo,
        rsp,
        spawn_drone,
        bridge,
        navigation,
        moving,
    ])