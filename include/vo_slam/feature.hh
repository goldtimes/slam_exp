#pragma once
#include <opencv2/features2d.hpp>
#include "vo_slam/common_include.hh"

namespace vo {

struct Frame;
struct MapPoint;
/**
 * @brief 2d特征点
 * 三角化之后会关联到一个mappoint, 以及对应的帧上
 */
struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Feature>;

    std::weak_ptr<Frame> frame_;         // 因为要和frame关联在一起,所以用weak_ptr
    cv::KeyPoint position_;              // 2d特征点位置
    std::weak_ptr<MapPoint> map_point_;  // 关联地图点

    bool is_outlier_ = false;       // 是否为异常点
    bool is_on_left_image_ = true;  // 标识提示在左图,false 在右图

    Feature() {
    }
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint& kp) : frame_(frame), position_(kp) {
    }
};
}  // namespace vo