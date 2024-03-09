cmake_minimum_required(VERSION 3.0.2)
project(vo_frontend)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

list(APPEND CMAKE_MODULE_PATH /home/hang/robot_algo_ws/src/slam_exp/src/vio/vo_frontend/cmake)

find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)

# find_package(OpenCV REQUIRED)

include_directories(
    include
    ${Sophus_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
)
add_executable(orb_cv orb_cv.cc)
target_link_libraries(orb_cv 
    ${OpenCV_LIBRARIES}
)

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cc)
target_link_libraries(pose_estimation_2d2d 
    ${OpenCV_LIBRARIES}
)

add_executable(triangulation triangulation.cc)
target_link_libraries(triangulation 
    ${OpenCV_LIBRARIES}
)

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cc)
target_link_libraries(pose_estimation_3d2d 
    ${OpenCV_LIBRARIES}
    ${Sophus_LIBRARIES}
    g2o_core
    g2o_stuff
)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cc)
target_link_libraries(pose_estimation_3d3d 
    ${OpenCV_LIBRARIES}
    ${Sophus_LIBRARIES}
    g2o_core
    g2o_stuff
)