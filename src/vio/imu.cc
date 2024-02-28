#include "vio/imu.hh"

#include <random>
#include "vio/utilities.hpp"

namespace vio{
    
    IMU::IMU(SensorParams params) : param_(params){
        gyro_bias_ = Eigen::Vector3d::Zero();
        acc_bias_ = Eigen::Vector3d::Zero();
    }
    
}