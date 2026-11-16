# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
import os

from setuptools import setup, find_packages

package_name = "d360"
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join(share_dir, "launch"), ['launch/d360_launch.py', 'launch/d360_min_launch.py']),
        # (os.path.join(share_dir, "checkpoints"), [os.path.join(package_name, "checkpoints", "contact_classifier.pth")]),
    ],
    entry_points={
        'console_scripts': [
            "serial_io = d360.d360_serial_io:main",
            "image_pub_opencv = d360.d360_image_pub:main",
            "image_pub_encode = d360.d360_image_encode:main",
            "audio_pub = d360.d360_audio_pub:main",
            "gas_pub = d360.d360_gas_pub:main",
            "classifier = d360.d360_classifier:main",
        ],
    },
)
