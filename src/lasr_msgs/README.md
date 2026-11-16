# lasr_msgs

Custom ROS 2 message and service definitions for the LASR grasping system.

## Messages

| Message | Description |
|---------|-------------|
| `Action` | Combined action command: hand command + xArm command |
| `Audio` | Audio data from tactile sensors |
| `ContactClassification` | Contact classification result (in contact + confidence) |
| `Finger` | Single finger state (4 joints) |
| `FingerPosition` | Finger position from VR tracking |
| `HandPosition` | Full hand position from VR tracking (thumb + 3 fingers) |
| `Qpos` | Joint positions (generalized) |
| `Qvel` | Joint velocities (generalized) |
| `RobotHand` | Hand pose request (left/right flag + thumb + 3 fingers) |
| `RobotHandEE` | Hand end-effector state |
| `XArmCommand` | xArm Cartesian/movement command |

## Services

| Service | Description |
|---------|-------------|
| `ClassifyD360Image` | Classify a Digit 360 image for contact detection (request: `Image` → response: `bool in_contact`, `float32 confidence`) |

## Dependencies

- `geometry_msgs`
- `std_msgs`
- `visualization_msgs`
- `builtin_interfaces`
- `rosidl_default_generators` (build)
- `rosidl_default_runtime` (runtime)

## License

MIT
