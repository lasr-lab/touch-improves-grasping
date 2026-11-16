
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from dataclasses import asdict
from d360.digit360.interface.usb.usb import get_digit360_devices

def generate_launch_description():
    digit360_devices = get_digit360_devices()

    if not digit360_devices:
        raise RuntimeError("No DIGIT360 devices found!")
    serial_list = [i.serial for i in digit360_devices]
    
    print(
        f"Found {len(serial_list)} DIGIT360 devices: {', '.join(serial_list)}\n"
    )

    device_infos = digit360_devices
    device_parameters = [asdict(device_info)for device_info in device_infos]
    ports = [i['data'] for i in device_parameters]
    video_devices = [i['ics'] for i in device_parameters]
    audio_devices = [i['audio'] for i in device_parameters]

    return LaunchDescription([
        # Node for serial_io
        Node(
            package="d360",
            executable="serial_io",
            name="serial_io",
            parameters=[
                {"ports": ports},
                {"reliable_publisher": False},
            ],
            output="screen",
            respawn=False,
            arguments=["--ros-args", "--log-level", "warn"],
            emulate_tty=True
        ),

        # Node for image publisher
        Node(
            package="d360",
            executable="image_pub_opencv",
            name="image_opencv_publisher_node",
            parameters=[
                {"devices": video_devices}
            ],
            output="screen",
            respawn=False,
            emulate_tty=True
        ),

        # Node for audio publisher
        Node(
            package="d360",
            executable="audio_pub",
            name="audio_publisher_node",
            parameters=[
                {"device_names": audio_devices}
            ],
            output="screen",
            respawn=False,
            emulate_tty=True
        ),
    ])
