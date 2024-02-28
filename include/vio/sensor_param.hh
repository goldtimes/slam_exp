#pragma once

#include <eigen3/Eigen/Core>

namespace vio {
class SensorParams {
   public:
    SensorParams(){
        Eigen::Matrix3d R;
        // 欧拉角 [ x: 1.5707963, y: 0, z: -1.5707963 ]
        R << 0,0,-1,
                -1,0,0,
                0,1,0;
        R_ic = R;
        t_ic = Eigen::Vector3d(0.05, 0.04, 0.03);
    }

    // sensor hz  
    int imu_hz = 200;
    int camera_hz = 30;
    // sensor dt
    double imu_timestep = 1.0  /  imu_hz;
    double camera_timestep = 1.0 / camera_hz;

    // bias noise 
    double gyro_bias_sigma = 0.00005;
    double acc_bias_sigma = 0.005;

    // noise
    double gyro_noise_sigma = 0.015;
    double acc_noise_sigma = 0.019;

    // camera noise 
    double pixel_noise = 1;

    // camera 焦距
    double fx = 460;
    double fy = 460;
    double cx = 255;
    double cy = 255;
    double image_width = 640;
    double image_height = 640;

    // T_C_to_I
    Eigen::Matrix3d R_ic;
    Eigen::Vector3d t_ic;
};
}  // namespace vio