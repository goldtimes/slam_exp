#pragma once
#include "vo_slam/common_include.hh"
#include "vo_slam/frame.hh"
#include "vo_slam/map.hh"

namespace vo {
class Map;

class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Backend>;

    Backend();

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
    }

   private:
    // 后端线程
    void BackendLoop();
    // 给定关键帧和路标进行优化
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr;
    Camera::Ptr cam_right_ = nullptr;
};
}  // namespace vo