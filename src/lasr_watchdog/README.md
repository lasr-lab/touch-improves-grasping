# lasr_watchdog

ROS 2 C++ watchdog node that monitors the freshness of sensor data during data collection and inference, publishing a health status flag.

## Overview

Subscribes to all critical sensor streams (Digit 360 images, RealSense images, joint states) and tracks the timestamp of the most recent message from each. A periodic timer compares ages against a configurable timeout and publishes a boolean status indicating whether all monitored sensors are actively reporting.

## Nodes

| Node | Description |
|------|-------------|
| `lasr_watchdog` | Monitors sensor data freshness and publishes health status |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_ms` | `int` | `1000` | Maximum age (ms) before a sensor is considered stale |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/digit_rgb/index_<0-3>` | `sensor_msgs/Image` | Digit tactile camera images |
| `/image_raw/index_<0-3>` | `sensor_msgs/Image` | Digit 360 raw images |
| `/xarm/joint_states` | `sensor_msgs/JointState` | xArm joint states |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/watchdog/status` | `std_msgs/Bool` | `true` if all monitored sensors are fresh, `false` if any are stale |

## Dependencies

- `rclcpp`
- `std_msgs`
- `sensor_msgs`
- [`lasr_msgs`](../lasr_msgs)

## License

MIT
