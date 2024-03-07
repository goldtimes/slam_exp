cmake_minimum_required(VERSION 3.0.2)
project(vo_frontend)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

# find_package(OpenCV REQUIRED)

# include_directories(
#     include
#     ${OpenCV_INCLUDE_DIRS}
# )
add_executable(orb_cv orb_cv.cc)
target_link_libraries(orb_cv 
    ${OpenCV_LIBRARIES}
)

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cc)
target_link_libraries(pose_estimation_2d2d 
    ${OpenCV_LIBRARIES}
)