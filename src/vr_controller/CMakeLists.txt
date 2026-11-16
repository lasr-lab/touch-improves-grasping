cmake_minimum_required(VERSION 3.16.3)
project(vr_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(lasr_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Add executables
add_executable(vr_controller
  src/vr_controller.cpp
)

# Target dependencies
ament_target_dependencies(vr_controller
  rclcpp
  lasr_msgs
)

target_include_directories(vr_controller PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

# Install targets
install(TARGETS
  vr_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
