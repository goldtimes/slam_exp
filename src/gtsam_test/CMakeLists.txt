cmake_minimum_required(VERSION 3.0.2)
project(gtsam_test)

find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time timer serialization)
find_package(GTSAM REQUIRED QUIET)

include_directories(
    include
    ${GTSAM_INCLUDE_DIR}
)
add_executable(odom_ex odom_ex.cc)

target_link_libraries(odom_ex
    gtsam
)