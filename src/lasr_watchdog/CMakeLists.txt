cmake_minimum_required(VERSION 3.16.3)
project(lasr_watchdog)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(lasr_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(lasr_watchdog
  src/lasr_watchdog.cpp
)

# Include directories
# include_directories(include)

# Add executables
add_executable(lasr_watchdog
  src/lasr_watchdog.cpp
)

# Target dependencies
ament_target_dependencies(lasr_watchdog
  rclcpp
  std_msgs
  sensor_msgs
  lasr_msgs
)

target_include_directories(${PROJECT_NAME} PUBLIC
${EIGEN3_INCLUDE_DIRS}
"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
"$<INSTALL_INTERFACE:include>"
)

# Install targets
install(TARGETS
  lasr_watchdog
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
