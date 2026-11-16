from setuptools import find_packages, setup

package_name = "image_udp_bridge"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NiklasLeukroth",
    maintainer_email="nleukroth@lasr.org",
    description="JSON serialization and UDP transmission for robot teleoperation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "udp_sender = image_udp_bridge.udp_sender:main",
        ],
    },
)
