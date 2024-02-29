#pragma once

#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <vector>
#include "vio/imu.hh"

inline Eigen::Matrix3d euler2Rotation(Eigen::Vector3d eulerAngles) {
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double yaw = eulerAngles(2);
    double cr = std::cos(roll);
    double sr = std::sin(roll);
    double cp = std::cos(pitch);
    double sp = std::sin(pitch);
    double cy = std::cos(yaw);
    double sy = std::sin(yaw);

    Eigen::Matrix3d Rib;  // body->惯性坐标系
    Rib << cy * cp, cy * sp * sr - sy * cr, sy * sr + cy * cr * sp, sy * cp, cy * cr + sy * sr * sp,
        sp * sy * cr - cy * sr, -sp, cp * sr, cp * cr;
    return Rib;
}

inline Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles) {
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);

    double cr = cos(roll);
    double sr = sin(roll);
    double cp = cos(pitch);
    double sp = sin(pitch);

    Eigen::Matrix3d R;
    R << 1, 0, -sp, 0, cr, sr * cp, 0, -sr, cr * cp;

    return R;
}
