# d360_msgs

ROS 2 message definitions for the [Digit 360](https://github.com/facebookresearch/digit360) modular tactile sensor platform.

## Messages

| Message | Description |
|---------|-------------|
| `AudioDataD360` | Raw audio sample data from a Digit 360 sensor |
| `AudioInfoD360` | Audio stream metadata (sample rate, channels, etc.) |
| `GasHTData` | Gas sensor temperature and humidity readings |
| `ImuEulerD360` | IMU data in Euler angle representation |
| `ImuQuatD360` | IMU data in quaternion representation |
| `ImuRawD360` | Raw IMU sensor readings |
| `LightControl` | LED/light control commands for Digit 360 illumination |
| `PressureAPD360` | Analog pressure sensor readings |
| `PressureD360` | Processed pressure data from the tactile sensor |

## Dependencies

- `geometry_msgs`
- `std_msgs`
- `rosidl_default_generators` (build)
- `rosidl_default_runtime` (runtime)

## License

CC BY-NC 4.0 — Copyright (c) Meta Platforms, Inc. and affiliates.
