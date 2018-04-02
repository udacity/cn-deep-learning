cmake_minimum_required(VERSION 2.8.3)
project(quad_controller_rl)

###########################
## General configuration ##
###########################

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

##################
## Catkin Tools ##
##################

## Find catkin macros and libraries
find_package(catkin REQUIRED
  message_generation
  std_msgs
  geometry_msgs
  std_srvs
)

##################
## Python setup ##
##################

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Add messages to generate, defined in the 'msg' folder
# add_message_files(
#   FILES
#   EulerAngles.msg
# )

## Add services to generate, defined in the 'srv' folder
add_service_files(
  FILES
  SetPose.srv
)

## Add actions to generate, defined in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
# )

## Generate added messages and services, with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_srvs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/params.cfg
# )

#################################
## Generate CMake config files ##
#################################
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  DEPENDS message_runtime
  DEPENDS std_msgs
  DEPENDS geometry_msgs
  DEPENDS std_srvs
)

###########
## Build ##
###########

## Specify additional header file locations
## Declare C++ libraries/executables
## Add dependencies/links for above libraries/executables

#############
## Install ##
#############

## Note: All install targets should use catkin DESTINATION variables
## See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

#############
## Testing ##
#############

## Add folders to be run by python nosetests
# catkin_add_nosetests(tests)
