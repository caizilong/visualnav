"""Optional ROS 2 camera, joystick and Twist mux; base driver is robot-specific."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def generate_launch_description():
    config = Path(__file__).resolve().parents[1] / "config"
    camera = yaml.safe_load((config / "camera_front.yaml").read_text())
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="false"),
        DeclareLaunchArgument("joystick", default_value="false"),
        DeclareLaunchArgument("mux", default_value="false"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        Node(package="usb_cam", executable="usb_cam_node_exe", name="usb_cam",
             parameters=[camera], remappings=[("image_raw", "/usb_cam/image_raw")],
             condition=IfCondition(LaunchConfiguration("camera"))),
        Node(package="joy", executable="joy_node", name="joy_node",
             condition=IfCondition(LaunchConfiguration("joystick"))),
        Node(package="twist_mux", executable="twist_mux", name="twist_mux",
             parameters=[str(config / "cmd_vel_mux.yaml")],
             remappings=[("cmd_vel_out", LaunchConfiguration("cmd_vel_topic"))],
             condition=IfCondition(LaunchConfiguration("mux"))),
    ])
