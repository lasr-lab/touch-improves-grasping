cmake_minimum_required(VERSION 3.10)
project(tilburg_hand_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ROS_DISABLE_TRACING ON)
add_definitions(-DTRACETOOLS_DISABLED)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Try to find dynamixel_sdk as ROS2 package first
find_package(dynamixel_sdk QUIET)
if(NOT dynamixel_sdk_FOUND)
  # If ROS2 package not found, try to find system installation
  find_path(DYNAMIXEL_SDK_INCLUDE_DIR
    NAMES dynamixel_sdk/dynamixel_sdk.h
    PATHS /usr/local/include /usr/include
    PATH_SUFFIXES dynamixel_sdk
  )
  
  find_library(DYNAMIXEL_SDK_LIBRARY
    NAMES dynamixel_sdk libdynamixel_sdk dxl_x64_cpp libdxl_x64_cpp
    PATHS /usr/local/lib /usr/lib
  )
  
  if(DYNAMIXEL_SDK_INCLUDE_DIR AND DYNAMIXEL_SDK_LIBRARY)
    message(STATUS "Found Dynamixel SDK: ${DYNAMIXEL_SDK_LIBRARY}")
    set(DYNAMIXEL_SDK_FOUND TRUE)
  else()
    message(FATAL_ERROR "Dynamixel SDK not found.")
  endif()
endif()

# Include directories
include_directories(include)

# Add executables
add_executable(tilburg_hand_interface
  src/tilburg_hand_interface.cpp
  src/motor_controller.cpp
)

# Target dependencies
ament_target_dependencies(tilburg_hand_interface
  rclcpp
  std_msgs
  sensor_msgs
)

# Link Dynamixel SDK
if(dynamixel_sdk_FOUND)
  # ROS2 package found
  ament_target_dependencies(tilburg_hand_interface dynamixel_sdk)
else()
  # System library found
  target_include_directories(tilburg_hand_interface PRIVATE ${DYNAMIXEL_SDK_INCLUDE_DIR})
  target_link_libraries(tilburg_hand_interface ${DYNAMIXEL_SDK_LIBRARY})
endif()

# Install targets
install(TARGETS
  tilburg_hand_interface
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
