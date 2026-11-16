# docker

Docker development environment for the touch-improves-grasping workspace. Provides a reproducible ROS 2 Jazzy container with all dependencies pre-installed, including TensorRT, RealSense, MoveIt, Gazebo, and xArm support.

## Contents

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build: `base` (ROS 2 + system deps) and `dev` (user, Python packages, xArm, RealSense, Gazebo) |
| `run_docker.sh` | Launch script with display, USB, audio, and workspace volume passthrough (supports native Linux and WSL) |
| `configs/colcon_config.yaml` | Colcon build configuration |

## Base Image

Built on `rwthika/ros2-tensorrt:jazzy-ros-base-v25.08` with additional packages:

- ROS 2 Jazzy: Foxglove bridge, cv_bridge, vision_opencv, realsense2_camera, MoveIt, ros2_control, Gazebo
- Tools: gdb, mold, tmux, htop, ipython3, usbutils
- Audio: portaudio, alsa-utils, alsa-tools

## Dev Image Additions

- Python: torch, numpy, open3d, opencv-python, digit_interface, pynput, einops, h5py, dm_control, sounddevice
- xArm ROS 2 (cloned and built from source)
- librealsense (built from source)
- Gazebo (gz-jetty)
- Non-root `devuser` with sudo and hardware group access

## Usage

```bash
# Build
docker build --build-arg USERNAME=devuser -t lasr_grasp:dev -f Dockerfile --target dev .

# Run (native Linux)
./run_docker.sh

# Run (WSL — uses port forwarding instead of --network=host)
# Automatically detected by run_docker.sh
```

The workspace root is mounted at `/home/devuser/lasr_grasp_ws` inside the container.

## Notes

- On WSL, `--network=host` does not work; the script forwards port 8765 for Foxglove instead.
- USB devices (Dynamixel, sensors) are passed through via `/dev/bus/usb`.
- Audio devices are passed through via `/dev/snd`.
- Ensure the Docker context is set to `default` if you encounter networking or GUI issues.
