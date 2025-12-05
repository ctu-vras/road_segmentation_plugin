# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Czech Technical University in Prague

cmake_minimum_required(VERSION 3.20.2)
project(road_segmentation_plugin)

set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(depthai REQUIRED)
find_package(depthai_ros_driver REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

add_library(road_segmentation_plugin SHARED
  src/road_segmentation_plugin.cpp
  src/road_segmentation.cpp
  src/nn_wrapper_road_segmentation.cpp
)
target_include_directories(road_segmentation_plugin PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(road_segmentation_plugin PUBLIC
  ${depthai_ros_driver_TARGETS}
  rclcpp::rclcpp
)
target_link_libraries(road_segmentation_plugin PRIVATE
  pluginlib::pluginlib
)
target_compile_definitions(road_segmentation_plugin PRIVATE DEPTHAI_VERSION_MAJOR=${depthai_VERSION_MAJOR})

file(GLOB MODELS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} models/*.json.in)
foreach(MODEL ${MODELS})
  string(REPLACE ".in" "" MODEL_OUT "${MODEL}")
  message(STATUS "Installing model ${MODEL_OUT}")
  configure_file("${MODEL}" ${CMAKE_CURRENT_BINARY_DIR}/${MODEL_OUT} @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${MODEL_OUT} DESTINATION share/${PROJECT_NAME}/models)
endforeach()

install(DIRECTORY models DESTINATION share/${PROJECT_NAME} PATTERN *.blob)

install(TARGETS road_segmentation_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

pluginlib_export_plugin_description_file(depthai_ros_driver plugins.xml)

if(BUILD_TESTING)
  find_package(cras_lint REQUIRED)
  cras_lint_common()
  cras_lint_cpp()
endif()

ament_package()
