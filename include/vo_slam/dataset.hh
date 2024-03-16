#pragma once

#include "vo_slam/camera.hh"
#include "vo_slam/common_include.hh"
#include "vo_slam/frame.hh"

namespace vo {
/**
 * @brief 数据集,读取相机和下个图像
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Dataset>;
    Dataset(const std::string& dataset_path);

    bool Init();
    Frame::Ptr NextFrame();

    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

   private:
    std::string dataset_path_;
    int current_image_index_ = 0;
    std::vector<Camera::Ptr> cameras_;
};
}  // namespace vo
