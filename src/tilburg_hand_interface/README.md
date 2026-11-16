# tilburg_hand_interface

ROS 2 C++ hardware interface for the [Tilburg Hand](https://www.tilburg-robotics.eu/), a 16-DOF dexterous robotic hand. Wraps the Tilburg Hand Python library and communicates with Dynamixel servos via the DynamixelSDK.

## Overview

Provides a ROS 2 node that receives joint target commands and drives the physical Tilburg Hand hardware. Handles per-side calibration (left/right), joint position mapping between degrees and Dynamixel ticks, and joint state feedback.

## Nodes

| Node | Description |
|------|-------------|
| `tilburg_hand_interface` | Main interface node — subscribes to joint targets, drives Dynamixel servos, publishes joint states |

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `side` | `string` | `"left"` or `"right"` — which hand this node controls |
| `usb-port` | `string` | Serial port for the Dynamixel USB-to-serial adapter (e.g., `/dev/ttyACM0`) |
| `joints-post-rate` | `int` | Rate (Hz) for publishing joint state feedback |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/th_<side>/joint_targets` | `std_msgs/Float64MultiArray` | Target joint positions (16 joints) |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/th_<side>/joint_states` | `sensor_msgs/JointState` | Current joint positions from encoder feedback |

## Architecture

- **`motor_controller.cpp`** — Low-level Dynamixel motor controller with per-side calibration tables mapping degrees ↔ ticks. Handles port initialization, servo communication, and shutdown.
- **`tilburg_hand_interface.cpp`** — ROS 2 node wrapping the motor controller, handling topic subscriptions and joint state publishing.

## Dependencies

- `rclcpp`
- `std_msgs`
- `sensor_msgs`
- `dynamixel_sdk` ([DynamixelSDK](../DynamixelSDK) submodule)

## License

MIT
