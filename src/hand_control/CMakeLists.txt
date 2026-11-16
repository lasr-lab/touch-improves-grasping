cmake_minimum_required(VERSION 3.16.3)
project(hand_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(lasr_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(hand_control src/hand_control.cpp)

ament_target_dependencies(hand_control
  rclcpp
  std_msgs
  lasr_msgs
  visualization_msgs
)

target_include_directories(hand_control PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

# Install targets
install(TARGETS
  hand_control
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
