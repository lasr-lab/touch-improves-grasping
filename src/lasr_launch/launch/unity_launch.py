import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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
    ip = LaunchConfiguration('ip')
    port = LaunchConfiguration('port')

    nodeCompressImage = Node(
            package='lasr_launch',
            executable='image_compression',
            parameters=[],
            output="screen",
            emulate_tty=True
        )

    nodeVRController = Node(
            package='vr_controller',
            executable='vr_controller',
            parameters=[],
            output="screen",
            emulate_tty=True)

    nodeUPDBridge = Node(
                package='image_udp_bridge',
                executable='udp_sender',
                parameters=[{"target_ip": ip, "target_port": port}],
                output="screen",
                emulate_tty=True)



    return LaunchDescription([
        DeclareLaunchArgument('ip'),
        DeclareLaunchArgument('port'),
        nodeCompressImage, nodeVRController, nodeUPDBridge])

