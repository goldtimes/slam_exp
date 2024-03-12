#pragma once

#include "common_include.hh"

namespace vo {
class Camera {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Camera>;
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;  // 相机内参
    SE3 pose_;                                                 // 外参
    SE3 pose_inv_;

    Camera() = default;
    Camera(double fx, double fy, double cx, double cy, double baseline, const SE3& pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose;
    }

    SE3 GetPose() const {
        return pose_;
    }

    Mat33 GetK() const {
        Mat33 K;
        K << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return K;
    }
    //  坐标变化
    //  p_c = T_c_w * p_w
    Vec3 world2camera(const Vec3& p_w, const Sophus::SE3& T_c_w);
    // p_w = T_c_w * p_c
    Vec3 camera2world(const Vec3& p_c, const SE3& T_c_w);

    Vec2 camera2pixel(const Vec3& p_c);

    /**
     * @brief 像素坐标到camera坐标
     * @param p_p 点在像素的坐标
     */
    Vec3 pixel2camera(const Vec2& p_p, double depth = 1);

    Vec3 pixel2world(const Vec2& p_p, const SE3& T_c_w, double depth = 1);

    Vec2 world2pixel(const Vec3& p_w, const SE3& T_c_w);
};
}  // namespace vo