cmake_minimum_required(VERSION 3.0.2)
project(relocalization)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system filesystem)

find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.10 REQUIRED QUIET)


FIND_PACKAGE(OpenMP REQUIRED) 
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "----------------FOUND OpenMP")
endif()

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES pcl_sac_ia
 CATKIN_DEPENDS roscpp pcl_conversions
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/pcl_sac_ia.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(sac_ia src/sac_ia.cc)

add_executable(scan_context_node 
  src/scan_context_node.cc
  src/Scancontext.cpp
  )


add_executable(pcd_to_bird
 src/pcd_to_bird.cc
)

add_executable(ndt_node
 src/ndt.cc
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(sac_ia
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  ${Boost_LIBRARIES}
)

target_link_libraries(scan_context_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(pcd_to_bird
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(ndt_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

