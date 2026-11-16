cmake_minimum_required(VERSION 3.8)
project(lasr_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  ${generic_msgs}
  "msg/Finger.msg"
  "msg/FingerPosition.msg"
  "msg/RobotHand.msg"
  "msg/HandPosition.msg"
  "msg/XArmCommand.msg"
  "msg/Qpos.msg"
  "msg/Qvel.msg"
  "msg/Action.msg"
  "msg/RobotHandEE.msg"
  "msg/Audio.msg"
  "msg/ContactClassification.msg"
  "srv/ClassifyD360Image.srv"


  DEPENDENCIES
   geometry_msgs
   visualization_msgs
   std_msgs
   sensor_msgs
)


ament_export_dependencies(std_msgs geometry_msgs rosidl_default_runtime visualization_msgs)
ament_export_include_directories(
  include
)

ament_package()
