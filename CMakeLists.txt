cmake_minimum_required(VERSION 3.0.2)
project(frustum_pointnet)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  vision_msgs
  ros_numpy
  message_generation
  darknet_ros_msgs
)

# Use if there is a setup.py folder
catkin_python_setup()

# add_message_files(
#   FILES
#   BoundingBox2D.msg
#   BoundingBox2DArray.msg
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   vision_msgs
#   frustum_pointnet
# )


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
  nodes/fpointnet_sub_node
  nodes/kitti_pub_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)