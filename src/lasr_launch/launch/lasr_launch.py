import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def get_config_file_path(name, dir='launch/params'):
    ret = os.path.join(
        get_package_share_directory('lasr_launch'),
        dir,
        name
    )
    return ret

def generate_launch_description():

    hand_control = Node(
        package='hand_control',
        executable='hand_control',
        parameters=[],
        output="screen",
        emulate_tty=True)

    nodeRecordBag = Node(
        package='lasr_launch',
        executable='ros_recorder',
        parameters=[],
        output="screen",
        emulate_tty=True
    )

    realsense2_camera_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[],
        output="screen",
        emulate_tty=True)
    
    tilburg_config = get_config_file_path("tilburgParams.yaml")
    
    tilburg_hand_interface = Node(
        package='tilburg_hand_interface',
        executable='tilburg_hand_interface',
        parameters=[tilburg_config],
        output="screen",
        emulate_tty=True)
    
    watchdog = Node(
        package='lasr_watchdog',
        executable='lasr_watchdog',
        parameters=[],
        output="screen",
        emulate_tty=True)

    classifier = Node(
            package='d360',
            executable='classifier',
            parameters=[],
            output="screen",
            emulate_tty=True)

    return LaunchDescription([
        hand_control, tilburg_hand_interface, realsense2_camera_node, nodeRecordBag, watchdog])

