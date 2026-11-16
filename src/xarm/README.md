# xarm

ROS 2 Python package for controlling the xArm 7 robot arm. Provides a movement node and launch files for arm initialization and teleoperation.

## Nodes

| Node | Source | Description |
|------|--------|-------------|
| `xarm_mover_node` | `xarm/xarm_mover_node.py` | Subscribes to `XArmCommand` messages and translates them into xArm service calls (cartesian moves, state changes, error resets) |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/xarm/command` | `lasr_msgs/XArmCommand` | Cartesian movement commands for the arm |
| `/xarm/robot_states` | `xarm_msgs/RobotMsg` | Current arm state feedback |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/xarm/reset` | `std_msgs/Bool` | Reset signal for arm error recovery |

### Services Used

| Service | Type | Description |
|---------|------|-------------|
| `/xarm/set_position` | `xarm_msgs/MoveCartesian` | Move to cartesian pose |
| `/xarm/set_state` | `xarm_msgs/SetInt16` | Set robot state (enable, disable, etc.) |
| `/xarm/set_mode` | `xarm_msgs/SetInt16` | Set robot mode |
| `/xarm/motion_enable` | `xarm_msgs/SetInt16ById` | Enable/disable individual joints |
| `/xarm/clean_error` | `xarm_msgs/Call` | Clear arm errors |

## Launch Files

| File | Description |
|------|-------------|
| `xarm_launch.py` | Launches the xArm 7 driver and mover node; configures robot IP, mode, collision sensitivity, and tool length |

### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_ip` | `192.168.1.245` | IP address of the xArm control box |
| `mode` | `7` | Robot mode (only mode 7 supports rapid position changes) |
| `collision_sensitivity` | `0` | Collision detection sensitivity (0 = off, 5 = most sensitive) |
| `tool_length` | `0.0` | Tool length in mm for tooltip rotation compensation |

## Dependencies

- `rclpy`
- `std_msgs`
- [`lasr_msgs`](../lasr_msgs)
- `xarm_msgs` (from [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2))

## License

MIT
