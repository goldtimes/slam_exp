#pragma once
#include "vo_slam/common_include.hh"
#include "vo_slam/frame.hh"
#include "vo_slam/map_point.hh"

namespace vo {
/**
 * @brief 地图
 * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Map>;
    using LandmarksType = std::unordered_map<unsigned long, MapPoint::Ptr>;
    using KeyframesType = std::unordered_map<unsigned long, Frame::Ptr>;

    Map() {
    }

    void InsertKeyframe(Frame::Ptr frame);

    void InsertMapPoint(MapPoint::Ptr map_point);

    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        // return
    }

    /// 获取所有关键帧
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// 获取激活地图点
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// 获取激活关键帧
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /**
     * @brief 清理map中观测数量为零的点
     */
    void CleanMap();

   private:
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;
    LandmarksType active_landmarks_;
    KeyframesType keyframes_;
    KeyframesType active_keyframes_;

    Frame::Ptr current_frame_ = nullptr;

    int num_active_keyframes_ = 7;  // 激活的关键帧数量
};
}  // namespace vo