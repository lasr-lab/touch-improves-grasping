# lasr_data_collector

ROS 2 C++ node that gathers data from all relevant sensor topics and republishes them time-synchronized for recording.

## Overview

Acts as a synchronization hub for multi-sensor data collection. Subscribes to arm state, hand state, camera images, Digit 360 tactile images, contact classifications, audio, and command topics. Uses a validity bitmask to track which sensors have fresh data and publishes a synchronized snapshot at a fixed interval (30 ms) only when all required sensors have reported.

## Nodes

| Node | Description |
|------|-------------|
| `lasr_data_collector` | Collects and time-synchronizes data from all sensors, republishes on `/act/*` topics |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/controls/hand/request` | `lasr_msgs/RobotHand` | Current hand command |
| `/xarm/command` | `lasr_msgs/XArmCommand` | Current arm command |
| `/xarm/joint_states` | `sensor_msgs/JointState` | Arm joint states |
| `/hand/joint_states` | `sensor_msgs/JointState` | Hand joint states |
| `/camera/color/image_raw` | `sensor_msgs/CompressedImage` | RealSense RGB image |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | RealSense depth image |
| `/image_raw/index_<0-3>` | `sensor_msgs/Image` | Digit 360 tactile images (4 sensors) |
| `/digit360/classification_<0-3>` | `lasr_msgs/ContactClassification` | Contact classification per sensor |
| `/d360/audio_<0-3>` | `d360_msgs/AudioDataD360` | Audio data per sensor |

## Published Topics

All republished under the `/act/` namespace with synchronized timestamps.

## Design

Uses a bitmask (`SensorBits`) to track data freshness. Each callback sets its bit; the publish timer checks that all required bits are set before emitting. Sensors can be selectively enabled/disabled by adjusting the `required_flags` mask.

## Dependencies

- `rclcpp`
- `std_msgs`, `sensor_msgs`, `geometry_msgs`
- [`lasr_msgs`](../lasr_msgs)
- [`d360_msgs`](../d360_msgs)

## License

MIT
