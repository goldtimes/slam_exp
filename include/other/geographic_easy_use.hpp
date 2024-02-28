#pragma once
#include <Eigen/Dense>
#include <GeographicLib/LocalCartesian.hpp>

namespace eskf {
/// @brief GNSS 数据类型
struct GNSSDataType {
    /// @brief 经度
    double longitude;
    /// @brief 纬度
    double latitude;
    /// @brief 高度
    double altitude;
    /// @brief 时间戳
    double stamp;
    /// @brief 状态，参考ros nav_fix的状态信息
    int status;
    /// @brief 方差矩阵
    Eigen::Vector3d vel;
    Eigen::Matrix3d cov;
    GNSSDataType() : longitude(0.0), latitude(0.0), altitude(0.0), stamp(0.0) {
    }
    /// @brief 构造函数，经纬高时间戳
    /// @param la_ 纬度
    /// @param lo_ 经度
    /// @param al_ 高度
    /// @param stamp_ 时间戳
    GNSSDataType(double la_, double lo_, double al_, double stamp_ = 0)
        : longitude(lo_), latitude(la_), altitude(al_), stamp(stamp_) {
    }
};
// 使用GeographicLib库将gnss坐标进行变换
class EZGeographic {
   private:
    GNSSDataType init_pose;
    bool pos_inited = false;
    GeographicLib::LocalCartesian geo_converter;

   public:
    EZGeographic(/* args */){};
    ~EZGeographic(){};
    EZGeographic(const GNSSDataType& gnss) : init_pose(gnss) {
        geo_converter.Reset(init_pose.latitude, init_pose.longitude, init_pose.altitude);
        pos_inited = true;
    }
    bool is_inited() {
        return pos_inited;
    }

    /// @brief 获取相对原点的东北天坐标
    /// @param gnss GNSS信息
    /// @param pos 笛卡尔坐标
    /// @return 未初始化时，转换会失败return false
    bool get_relative_position(const GNSSDataType& gnss, Eigen::Vector3d& pos) {
        if (!pos_inited) {
            return false;
        }
        geo_converter.Forward(gnss.latitude, gnss.longitude, gnss.altitude, pos.x(), pos.y(), pos.z());
        return true;
    }

    /// @brief 重新设置原点
    /// @param gnss_ 原点GNSS信息
    void reset_gnss_pose(const GNSSDataType& gnss_) {
        init_pose = gnss_;
        geo_converter.Reset(init_pose.latitude, init_pose.longitude, init_pose.altitude);
        pos_inited = true;
    }
};
}  // namespace eskf