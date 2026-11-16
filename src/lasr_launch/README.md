# lasr_launch

Central launch files and configurations for the LASR grasping system. Orchestrates all nodes across the workspace.

## Launch Files

| File | Description |
|------|-------------|
| `lasr_launch.py` | Main system launch — starts hand control, data collector, RealSense camera, Digit 360 sensors, watchdog, and Tilburg hand interfaces |
| `unity_launch.py` | VR teleoperation launch — starts image compression, VR controller, and UDP bridge to Unity |
| `foxglove.launch.py` | Foxglove bridge for remote visualization and debugging |

## Usage

```bash
# Full system
ros2 launch lasr_launch lasr_launch.py

# VR teleoperation
ros2 launch lasr_launch unity_launch.py ip:=<unity_ip> port:=<unity_port>

# Foxglove visualization
ros2 launch lasr_launch foxglove.launch.py
```

## Configuration

Launch parameters are stored in `launch/params/`:

| File | Description |
|------|-------------|
| `tilburgParams.yaml` | Tilburg Hand USB port and joint configuration |

## Nodes Launched

The main launch file starts:
- `hand_control` — hand kinematics and joint target computation
- `lasr_data_collector` — synchronized data recording
- `realsense2_camera_node` — Intel RealSense camera driver
- Digit 360 sensor nodes
- `lasr_watchdog` — sensor health monitoring
- `tilburg_hand_interface` (left + right) — Dynamixel hand drivers

## Dependencies

- `rclcpp`
- All packages in this workspace

## License

MIT
