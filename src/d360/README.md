# d360

ROS 2 Python driver for the [Digit 360](https://github.com/facebookresearch/digit360) modular tactile sensor platform developed by Meta. Captures images, audio, gas, and IMU data from multiple Digit 360 sensors and publishes them as ROS 2 topics.

## Nodes

| Node | Source | Description |
|------|--------|-------------|
| `d360_image_pub` | `d360/d360_image_pub.py` | Captures images from Digit 360 cameras and publishes raw + compressed `Image` topics per sensor |
| `d360_image_encode` | `d360/d360_image_encode.py` | Image encoding/compression utilities |
| `d360_audio_pub` | `d360/d360_audio_pub.py` | Publishes audio data from Digit 360 sensors |
| `d360_gas_pub` | `d360/d360_gas_pub.py` | Publishes gas sensor readings |
| `d360_classifier` | `d360/d360_classifier.py` | Contact classification node using a CNN classifier on Digit 360 images; exposes a `ClassifyD360Image` service |
| `d360_serial_io` | `d360/d360_serial_io.py` | Low-level serial I/O communication with Digit 360 hardware |

## Published Topics

- `/image_raw/index_<i>` (`sensor_msgs/Image`) — raw camera images from each Digit 360 sensor
- Compressed image variants for bandwidth-efficient transport
- Audio, gas, and IMU topics as defined in [`d360_msgs`](../d360_msgs)

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `devices` | `string_array` | List of device paths for connected Digit 360 sensors |

## Configuration

Sensor configuration is stored in `config/hand_config.yaml`. Launch files are provided in `launch/`:

- `d360_launch.py` — full launch with all sensors
- `d360_hand_launch.py` — launch with hand integration
- `d360_min_launch.py` — minimal launch for testing

## Dependencies

- `rclpy`
- [`d360_msgs`](../d360_msgs)
- `cv_bridge`, `OpenCV`
- `digit_interface`

## License

CC BY-NC 4.0 — Copyright (c) Meta Platforms, Inc. and affiliates.
