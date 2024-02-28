#pragma once
#include <Eigen/Dense>

namespace eskf {
Eigen::Matrix3d skew(const Eigen::Vector3d& value);
// so3指数->so3李群的映射
Eigen::Matrix3d so3Exp(const Eigen::Vector3d& so3);
// so3李群->so3指数的映射
Eigen::Vector3d SO3Log(const Eigen::Matrix3d& R);
}  // namespace eskf