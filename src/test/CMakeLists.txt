cmake_minimum_required(VERSION 3.0.2)
project(test)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(eigen_test eigen_test.cc)
target_link_libraries(eigen_test)