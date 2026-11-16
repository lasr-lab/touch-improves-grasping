cmake_minimum_required(VERSION 3.16.3)
project(lasr_data_collector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(d360_msgs REQUIRED)
find_package(lasr_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(lasr_data_collector
  src/lasr_data_collector.cpp
)

# Include directories
# include_directories(include)

# Add executables
add_executable(lasr_data_collector
  src/lasr_data_collector.cpp
)

# Target dependencies
ament_target_dependencies(lasr_data_collector
  rclcpp
  std_msgs
  sensor_msgs
  d360_msgs
  lasr_msgs
)

target_include_directories(${PROJECT_NAME} PUBLIC
${EIGEN3_INCLUDE_DIRS}
"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
"$<INSTALL_INTERFACE:include>"
)

# Install targets
install(TARGETS
  lasr_data_collector
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
