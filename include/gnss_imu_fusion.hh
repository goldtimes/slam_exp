#pragma once

#include <deque>
#include "geographic_easy_use.hpp"
#include "lib/ESKF/eskf.hh"

// GPS观测数据
struct Obs {
    double stamp;
    eskf::GNSSDataType gnss;
    // yaw
};

struct CachedItem {
    eskf::ESKF::State state;
    Eigen::Matrix<double, 18, 18> P;
    eskf::IMU imu;
    double imu_start_time;
    double imu_end_time;
};

class GNSSImuFusion {};