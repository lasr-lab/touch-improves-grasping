# unity

Unity project for VR-based teleoperation of the robotic grasping system. Provides a VR interface for hand tracking and robot control via the [ROS-TCP-Endpoint](../ROS-TCP-Endpoint).

## Overview

The Unity application connects to the ROS 2 system through the ROS-TCP Endpoint, receiving camera images and joint states while sending VR hand tracking data. It uses the ROS message browser to display custom message types defined in [`lasr_msgs`](../lasr_msgs) and [`d360_msgs`](../d360_msgs).

## Contents

| Directory | Description |
|-----------|-------------|
| `Assets/` | Unity assets including ROS message definitions, Oculus VR integration, input actions, and UI elements |
| `Assets/ROSMessages/` | Generated ROS message classes for Unity-side deserialization |
| `Assets/Oculus/` | Oculus VR SDK integration |
| `Assets/Plugins/` | Platform-specific plugins |
| `Packages/` | Unity package manifest |

## Integration

The Unity project communicates with ROS 2 via:

1. **ROS-TCP Endpoint** — receives ROS messages and forwards them to Unity
2. **UDP bridge** — [`image_udp_bridge`](../image_udp_bridge) sends compressed images and joint states over UDP
3. **VR Controller** — [`vr_controller`](../vr_controller) receives hand tracking data from Unity and converts it to hand commands

## Requirements

- Unity (version matching the project's `ProjectSettings`)
- Oculus/Meta Quest VR headset
- ROS-TCP Endpoint running on the ROS 2 side

## License

MIT
