# vr_controller

ROS 2 C++ node that translates VR hand tracking data into structured hand commands for the Tilburg Hand.

## Overview

Receives raw hand tracking poses from a VR headset (via Unity) and converts them into anatomically meaningful finger joint angles. Handles coordinate frame transformations, thumb rotation offsets, and joint limit clamping before publishing `RobotHand` messages for the [`hand_control`](../hand_control) node.

## Nodes

| Node | Description |
|------|-------------|
| `vr_controller` | Converts VR hand tracking to hand commands |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/vr/hand/request` | `lasr_msgs/HandPosition` | Raw hand tracking data from VR (thumb + 4 fingers) |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/controls/hand/request` | `lasr_msgs/RobotHand` | Processed hand command for `hand_control` |

## Design

- Thumb base rotation offset: 60°
- Thumb base joint offset: 40°
- Finger base rotation limit: ±20°
- Uses Eigen for coordinate transformations
- `Finger.hpp` contains the finger kinematic model

## Dependencies

- `rclcpp`
- [`lasr_msgs`](../lasr_msgs)
- Eigen3

## License

MIT
