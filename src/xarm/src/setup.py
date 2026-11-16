import os

from setuptools import setup

package_name = "xarm"
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join(share_dir, "launch"), ["launch/xarm_launch.py"])
    # ]
    ],
    entry_points={
        'console_scripts': [
            'xarm_mover_node = xarm.xarm_mover_node:main'
        ],
    },
)
