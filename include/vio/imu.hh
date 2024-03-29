#pragma once
#include "vio/sensor_param.hh"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>

namespace vio {
// 运动数据
struct MotionData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    Eigen::Vector3d imu_acc;
    Eigen::Vector3d imu_gyro;

    Eigen::Vector3d imu_gyro_bias;
    Eigen::Vector3d imu_acc_bias;

    Eigen::Vector3d imu_velocity;
};

class IMU {
   public:
    IMU(SensorParams param);
    MotionData MotionModel(double t);
    void addIMUMotionDataWithNoise(MotionData& data);
    void IntegrationIMU(std::string data_path, std::string traj_path);

   public:
    SensorParams param_;
    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d acc_bias_;

    Eigen::Vector3d init_velocity_;
    Eigen::Vector3d init_twb_;
    Eigen::Matrix3d init_Rwb_;
};
}  // namespace vio
