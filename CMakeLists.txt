cmake_minimum_required(VERSION 3.0.2)
project(manhole_detector)

add_definitions(-std=c++14)

find_package(catkin_simple REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(planner_common REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

catkin_package(
	INCLUDE_DIRS include
	CATKIN_DEPENDS roscpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs
	DEPENDS PCL EIGEN3
)


include_directories(
	${Eigen_INCLUDE_DIRS}
	include
)

cs_add_library(${PROJECT_NAME}
	src/manhole_detector.cpp
)

# Frontier Detector Node
cs_add_executable(manhole_detector_node src/manhole_detector_node.cpp)
target_link_libraries(manhole_detector_node ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

cs_add_executable(depth_img_publisher src/depth_img_publisher.cpp)
target_link_libraries(depth_img_publisher ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

cs_install()
cs_export(LIBRARIES)