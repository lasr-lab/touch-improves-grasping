# image_udp_bridge

ROS 2 Python package that forwards ROS topics to Unity over UDP for VR teleoperation. Handles JSON serialization, image compression, UDP transport with chunking, and rate-limited pacing.

## Overview

Bridges the ROS 2 ecosystem with a Unity-based VR interface by sending camera images, joint states, and other sensor data over UDP. Rate limiting and compression are handled externally; this node focuses on UDP transport, chunking large messages, and pacing chunks to avoid kernel-level bursting over WiFi.

## Nodes

| Node | Source | Description |
|------|--------|-------------|
| `udp_sender` | `image_udp_bridge/udp_sender.py` | Subscribes to ROS topics and forwards them as JSON-encoded UDP packets to a Unity client |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_ip` | `string` | *(required)* | IP address of the Unity client |
| `target_port` | `int` | *(required)* | UDP port on the Unity client |
| `chunk_delay_ms` | `double` | `2.0` | Delay between UDP chunks in milliseconds |

## Transport Details

- Maximum UDP payload: 40,000 bytes
- Send buffer: 128 KB
- Large messages are split into chunks with base64-encoded payloads
- Uses `sensor_msgs/CompressedImage` for bandwidth-efficient image transport

## Dependencies

- `rclpy`
- `sensor_msgs`
- `trajectory_msgs`
- `geometry_msgs`
- `builtin_interfaces`

## License

MIT
