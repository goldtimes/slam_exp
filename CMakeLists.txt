cmake_minimum_required(VERSION 3.0.2)
project(slam_exp)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.10 REQUIRED)
find_package(Ceres REQUIRED)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES slam_exp
#  CATKIN_DEPENDS roscpp
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
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/slam_exp.cpp
# )
add_subdirectory(src/vio/vo_frontend)
add_subdirectory(src/test)

add_library(vio_backend 
  src/vio/curve_fitting/backend/edge.cc
  src/vio/curve_fitting/backend/vertex.cc
  src/vio/curve_fitting/backend/problem.cc)

# target_link_libraries(vio_backend)

add_executable(sim_imu_data 
  src/vio/sim_imu_data.cc
  src/vio/imu.cc
  )
  ## Specify libraries to link a library or executable target against
target_link_libraries(sim_imu_data
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
)

add_executable(
  curve_fitting
  src/vio/curve_fitting/app/CurveFitting.cc)

target_link_libraries(curve_fitting
  vio_backend
  glog
)


add_executable(gaussNewton src/vio/curve_fitting/app/gaussNewton.cc)
target_link_libraries(gaussNewton)

add_executable(ceresCurveFitting src/vio/curve_fitting/app/ceresCurveFitting.cc)
target_link_libraries(ceresCurveFitting
  ${CERES_LIBRARIES}
)


