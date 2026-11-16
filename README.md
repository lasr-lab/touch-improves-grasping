# Touch Improves Learning Dexterous Grasping from Demonstration

This is the official code repository for the paper **"Touch Improves Learning Dexterous Grasping from Demonstration"** by Niklas Leukroth, Pankhuri Vanjani, Jakub Sulga, Ken Nakahara, Rudolf Lioutikov, and Roberto Calandra.

**Project Website:** https://lasr-lab.github.io/touch-improves-grasping/

---

A ROS 2 Jazzy workspace for research on tactile-improved robotic grasping. The system integrates Digit 360 tactile sensors, a Tilburg dexterous hand, an xArm 7, VR-based teleoperation, and learning-from-demonstration policies (BESO, ACT, Flow Matching) with optional tactile representations.

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ VR Hand      │──▶│vr_controller │──▶│ hand_control   │
│ Tracking     │   │(VR → hand    │   │(hand cmd →     │
│ (Unity)      │   │ commands)    │   │ joint targets) │
└─────────────┘   └──────────────┘   └───────┬────────┘
                                              │
┌─────────────┐   ┌──────────────┐   ┌───────▼────────┐
│ image_udp   │   │ lasr_data    │   │ tilburg_hand   │
│ _bridge     │   │ _collector   │   │ _interface     │
│(ROS↔Unity   │   │(sync & record│   │(Dynamixel servo│
│ UDP)        │   │ all topics)  │   │ driver)        │
└─────────────┘   └──────────────┘   └────────────────┘
                                              │
┌─────────────┐   ┌──────────────┐   ┌───────▼────────┐
│ d360        │   │ lasr_watchdog│   │ DynamixelSDK   │
│(Digit 360   │   │(sensor health│   │ (submodule)    │
│ driver)     │   │ monitor)     │   │                │
└─────────────┘   └──────────────┘   └────────────────┘

┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ inference   │   │ xarm         │   │ realsense-ros  │
│(policy      │   │(xArm 7 move  │   │ (submodule)    │
│ training &  │   │ commands)    │   │                │
│ deployment) │   │              │   │                │
└─────────────┘   └──────────────┘   └────────────────┘
```

## Packages

| Package | Type | Description |
|---------|------|-------------|
| [`d360`](src/d360) | Python | Digit 360 tactile sensor driver (image, audio, gas, IMU) |
| [`d360_msgs`](src/d360_msgs) | CMake | ROS 2 message definitions for Digit 360 sensors |
| [`docker`](src/docker) | Config | Docker development environment (ROS 2 Jazzy + TensorRT) |
| [`hand_control`](src/hand_control) | C++ | Converts hand requests to joint targets with RViz visualization |
| [`image_udp_bridge`](src/image_udp_bridge) | Python | UDP bridge forwarding ROS topics to Unity for teleoperation |
| [`inference`](src/inference) | Python | Policy training & deployment (BESO, ACT, Flow Matching + tactile variants) |
| [`lasr_data_collector`](src/lasr_data_collector) | C++ | Time-synchronized multi-sensor data collection and recording |
| [`lasr_launch`](src/lasr_launch) | Python | Central launch files for the full system |
| [`lasr_msgs`](src/lasr_msgs) | CMake | Custom ROS 2 message and service definitions |
| [`lasr_watchdog`](src/lasr_watchdog) | C++ | Watchdog monitoring sensor data freshness |
| [`scripts`](src/scripts) | Python | Dataset annotation, conversion, and video creation utilities |
| [`tilburg_hand_interface`](src/tilburg_hand_interface) | C++ | Hardware driver for the Tilburg Hand via Dynamixel servos |
| [`vr_controller`](src/vr_controller) | C++ | Translates VR hand tracking into hand commands |
| [`xarm`](src/xarm) | Python | xArm 7 movement commands and launch files |

### Git Submodules

| Submodule | Source |
|-----------|--------|
| `DynamixelSDK` | [ROBOTIS-GIT/DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) |
| `realsense-ros` | [realsenseai/realsense-ros](https://github.com/realsenseai/realsense-ros) |
| `ROS-TCP-Endpoint` | [Unity-Technologies/ROS-TCP-Endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint) |

## Prerequisites

- Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/)
- Python 3.10+
- Docker (optional, for containerized development)

## Building

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd touch-improves-grasping

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Install rosdeps
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build

# Source workspace
source install/setup.bash
```

### Docker

```bash
cd src/docker

# Build the dev image
docker build --build-arg USERNAME=devuser -t lasr_grasp:dev -f Dockerfile --target dev .

# Run (native Linux)
./run_docker.sh
```

## Running

Launch the full system:

```bash
ros2 launch lasr_launch lasr_launch.py
```

Launch with Unity/VR teleoperation:

```bash
ros2 launch lasr_launch unity_launch.py ip:=<unity_ip> port:=<unity_port>
```

Launch Foxglove bridge for visualization:

```bash
ros2 launch lasr_launch foxglove.launch.py
```

## Inference & Training

The `inference` package supports multiple policy families with and without tactile input:

- **BESO** / **BESO+Tactile** — diffusion-based policy
- **ACT** / **ACT+Tactile** — action chunking with transformers
- **Flow Matching** / **FM+Tactile** — flow matching policy

See [`src/inference/README.md`](src/inference/README.md) for details on training and deployment.

## License

Individual packages carry their own licenses. The Digit 360 components (`d360`, `d360_msgs`) are licensed under CC BY-NC 4.0 (Copyright Meta Platforms, Inc.). All other packages are licensed under the MIT License unless otherwise noted.
