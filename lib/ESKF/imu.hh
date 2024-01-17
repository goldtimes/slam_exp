#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace eskf {
class IMU {
   public:
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gro = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rot = Eigen::Quaterniond::Identity();
    double stamp = 0.0;  // s

    friend std::ostream& operator<<(std::ostream& ostream, const IMU& imu) {
        ostream << "imu_time: " << imu.stamp << " ms | imu_acc: " << imu.acc.transpose()
                << " | imu_gro: " << imu.gro.transpose();
        return ostream;
    }
};
}  // namespace eskf