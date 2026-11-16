# hand_control

ROS 2 C++ node that converts high-level hand requests (`RobotHand` messages) into joint-level target commands for the Tilburg Hand, with real-time RViz visualization.

## Overview

Subscribes to hand pose requests and translates them into per-finger joint targets using a kinematic hand model. Publishes joint commands to the Tilburg Hand interface and visualization markers for RViz.

## Nodes

| Node | Description |
|------|-------------|
| `hand_control` | Main node — receives `RobotHand` requests, computes joint targets via `HandModel`, publishes to hardware and visualization |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/controls/hand/request` | `lasr_msgs/RobotHand` | Desired hand pose (thumb + 3 fingers, left/right selection) |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/th_right/joint_targets` | `std_msgs/Float64MultiArray` | Joint targets for the right hand |
| `/th_left/joint_targets` | `std_msgs/Float64MultiArray` | Joint targets for the left hand |
| `/visualization/hand` | `visualization_msgs/MarkerArray` | RViz markers for hand visualization |

## Architecture

- **`HandModel.hpp`** — Kinematic hand model with bounding-box finger geometry, forward kinematics, and conversion between `RobotHand` messages and joint targets. Uses Eigen for transformations.
- **`hand_control.cpp`** — ROS 2 node wrapping the hand model, handling subscriptions and publishing.

## Dependencies

- `rclcpp`
- `std_msgs`
- `visualization_msgs`
- [`lasr_msgs`](../lasr_msgs)
- Eigen3

## License

MIT
