cmake_minimum_required(VERSION 3.0.2)
project(bb_pub_node)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  vision_msgs
  ros_numpy
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES bb_pub_node
#  CATKIN_DEPENDS rospy std_msgs vision_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS 
  ros/fpointnet_sub_node.py
  ros/kitti_pub_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)