#!/usr/bin/env python3
"""
launching the model detectpr node and the controller
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share  = get_package_share_directory('IGVC_lane_follow')
    params_file = str(Path(pkg_share) / 'config' / 'params.yaml')

    return LaunchDescription([
        # ── Camera node ───────────────────────────────────────────────────────
        Node(
            package    = 'lane_follower',
            executable = 'camera_node',
            name       = 'camera_node',
            output     = 'screen',
            parameters = [params_file],
        ),

        # ── Lane detector node ────────────────────────────────────────────────
        Node(
            package    = 'lane_follower',
            executable = 'detector_node',
            name       = 'detector_node',
            output     = 'screen',
            parameters = [params_file],
        ),

        # ── Lane controller node ──────────────────────────────────────────────
        Node(
            package    = 'lane_follower',
            executable = 'controller_node',
            name       = 'controller_node',
            output     = 'screen',
            parameters = [params_file],
        ),
    ])
