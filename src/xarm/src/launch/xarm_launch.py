import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnExecutionComplete
from launch.substitutions import FindExecutable
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

# config params
robot_ip = "192.168.1.245" # String containing IP on the backside of the control box
mode = 7 # int from 0 to 7 to set robot mode, !!!Only mode 7 works well with rapid change of position!!!
collision_sensitivity = 0 # int from 0 to 5, higher values cause robot to stop early when applying pressure
                          # !!!setting this to 0 completely shuts off collision detection, be careful!!! 
tool_length = 0.0 # length of the tool attached in mm, used to get rotations right around tooltip


def generate_launch_description():
   xarm_driver = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory("xarm_api"), "launch"),
         "/xarm7_driver.launch.py"]),
      launch_arguments={"robot_ip": robot_ip}.items(),
   )
   xarm_mover = Node(package="xarm", executable="xarm_mover_node", output="screen",
        emulate_tty=True)
   motion_enable = ExecuteProcess(
      cmd=[[
         FindExecutable(name="ros2"),
         " service call ",
         "/xarm/motion_enable ",
         "xarm_msgs/srv/SetInt16ById ",
         '"{id: 8, data: 1}"',
      ]],
      shell=True,
   )

   ld = LaunchDescription([
      xarm_driver,
      xarm_mover,
      # next part looks awful but is a powerful solution to call the "set_state" service 
      # after every other service has finished, since the robots state might not be 0 if another service 
      # has been called after "set_state" which causes the robot to be "not ready to move"
      RegisterEventHandler(
         # execute the TimerAction after the "motion_enable" service has completed
         OnExecutionComplete(
            target_action=motion_enable,
            on_completion=[
               # use TimerAction to execute "set_state" after 5 seconds to ensure all other services finished
               TimerAction(
                  period=5.0,
                  actions=[
                     # set robot state to 0 ("standby/ready to move")
                     ExecuteProcess(
                        cmd=[[
                           FindExecutable(name="ros2"),
                           " service call ",
                           "/xarm/set_state ",
                           "xarm_msgs/srv/SetInt16 ",
                           '"{data: 0}"',
                        ]],
                        shell=True,
                     )
                  ],
               )
            ]
         )
      ),
   ])

   # set collision sensitivity level to value from config params
   ld.add_action(
      ExecuteProcess(
         cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            "/xarm/set_collision_sensitivity ",
            "xarm_msgs/srv/SetInt16 ",
            f'"{{data: {int(collision_sensitivity)}}}"',
         ]],
         shell=True,
      )
   )

   # set Tool Center Point offset to value from config params 
   # to make the robot rotate around the tooltip
   ld.add_action(
      ExecuteProcess(
         cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            "/xarm/set_tcp_offset ",
            "xarm_msgs/srv/SetFloat32List ",
            f'"{{datas: [0,0,{float(tool_length)},0,0,0]}}"',
         ]],
         shell=True,
      )
   )

   # enable all joints
   ld.add_action(
      motion_enable
   )

   # clean errors
   ld.add_action(
      ExecuteProcess(
         cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            "/xarm/clean_error ",
            "xarm_msgs/srv/Call ",
         ]],
         shell=True,
      )
   )

   # set robot mode to value from config params
   ld.add_action(
      ExecuteProcess(
         cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            "/xarm/set_mode ",
            "xarm_msgs/srv/SetInt16 ",
            f'"{{data: {int(mode)}}}"',
         ]],
         shell=True,
      )
   )
   
   return ld
