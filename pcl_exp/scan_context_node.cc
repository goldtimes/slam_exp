#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "Scancontext.h"

using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;

struct MetaInfo {
    PointCloud::Ptr lidar_data;
    PointCloud::Ptr lidar_down;
    int level;
    std::string name;
    Eigen::Matrix4d T;
};

std::map<int, std::vector<std::shared_ptr<MetaInfo>>> meta_group;

PointCloud::Ptr traj_cloud(new PointCloud);
PointCloud::Ptr global_map_ptr_(new PointCloud);
pcl::VoxelGrid<PointType> voxelgrid_;
SCManager scmanager;

std::string padZeros(int val, int num_digits = 6) {
    std::ostringstream out;
    out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
    return out.str();
}

void saveSCD(std::string filename, Eigen::MatrixXd matrix, std::string delimiter = " ") {
    int precision = 3;  // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");

    std::ofstream file(filename);
    if (file.is_open()) {
        file << matrix.format(the_format);
        file.close();
    }
}

void load_pcd_and_make_context(const std::string& map_dir, int level, std::string& save_sc_dir) {
    int pcd_count = 0;
    voxelgrid_.setLeafSize(0.2, 0.2, 0.2);
    boost::filesystem::path dir(map_dir);
    if (boost::filesystem::exists(dir)) {
        for (const auto& iter : boost::filesystem::directory_iterator(dir)) {
            // 获取图元的目录
            if (boost::filesystem::is_directory(iter)) {
                std::stringstream ss;
                ss << iter.path().string() << "/data.yaml";
                std::string meta_name = iter.path().filename().string();
                std::cout << "load meta: " << meta_name << std::endl;
                std::cout << "load file: " << ss.str().c_str() << std::endl;
                if (!boost::filesystem::exists(ss.str())) {
                    std::cout << "config file  not exist, SKIP" << ss.str().c_str();
                    continue;
                }
                try {
                    YAML::Node config = YAML::LoadFile(ss.str());
                    // 解析楼层
                    int level = config["floor"].as<int>();
                    // 解析变换
                    std::vector<double> data = config["T"].as<std::vector<double>>();
                    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                    T.block<3, 3>(0, 0) = Eigen::Quaterniond(data[6], data[3], data[4], data[5]).toRotationMatrix();
                    T.block<3, 1>(0, 3) = Eigen::Vector3d(data[0], data[1], data[2]);
                    if (meta_group.find(level) == meta_group.end()) {
                        meta_group[level] = std::vector<std::shared_ptr<MetaInfo>>();
                    }
                    meta_group[level].push_back(std::make_shared<MetaInfo>());
                    auto& meta_info = meta_group[level].back();
                    meta_info->level = level;
                    meta_info->name = meta_name;
                    meta_info->lidar_data.reset(new PointCloud);
                    meta_info->lidar_down.reset(new PointCloud);
                    meta_info->T = T;

                    std::stringstream ss1;
                    ss1 << iter.path().string() << "/data_trajectory.pcd";
                    if (boost::filesystem::exists(boost::filesystem::path(ss1.str()))) {
                        PointCloud::Ptr tmp_cloud(new PointCloud);
                        PointCloud::Ptr trans_cloud(new PointCloud);
                        pcl::io::loadPCDFile<PointType>(ss1.str(), *tmp_cloud);
                        pcl::transformPointCloud(*tmp_cloud, *trans_cloud, T);
                        *traj_cloud += *trans_cloud;
                    } else {
                        std::cout << "no traj file: " << ss1.str() << std::endl;
                    }

                    // load pcd
                    std::stringstream ss2;
                    ss2 << iter.path().string() << "/data.pcd";
                    std::cout << "data file: " << ss2.str() << std::endl;
                    if (boost::filesystem::exists(boost::filesystem::path(ss2.str()))) {
                        std::cout << "load pcd count: " << ++pcd_count << std::endl;
                        pcl::io::loadPCDFile<PointType>(ss2.str(), *meta_info->lidar_data);
                        voxelgrid_.setInputCloud(meta_info->lidar_data);
                        voxelgrid_.filter(*meta_info->lidar_down);
                        scmanager.makeAndSaveScancontextAndKeys(*meta_info->lidar_down);
                        const auto& curr_scd = scmanager.getConstRefRecentSCD();
                        std::string curr_scd_node_idx = padZeros(scmanager.polarcontexts_.size() - 1);
                        saveSCD(save_sc_dir + curr_scd_node_idx + ".scd", curr_scd);
                        PointCloud::Ptr trans_cloud(new PointCloud);
                        pcl::transformPointCloud(*meta_info->lidar_data, *trans_cloud, T);
                        *global_map_ptr_ += *trans_cloud;
                    } else {
                        std::cout << "no data file: " << ss2.str() << std::endl;
                    }

                } catch (std::exception& e) {
                    std::cout << "load trajectory error: " << e.what() << std::endl;
                    continue;
                }
            } else {
                std::cout << "continu" << std::endl;
                continue;
            }
        }
    }
    if (!global_map_ptr_->empty()) {
        std::cout << "global map size: " << global_map_ptr_->size() << std::endl;
        voxelgrid_.setInputCloud(global_map_ptr_);
        voxelgrid_.filter(*global_map_ptr_);
        pcl::io::savePCDFileBinary("/home/kilox/global_map.pcd", *global_map_ptr_);
        std::cout << "after filter, global map size: " << global_map_ptr_->size() << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan_context");
    ros::NodeHandle nh;
    std::string map_dir = "/home/kilox/kilox_ws/wutongdao/meta_maps";
    std::string save_sc_dir = "/home/kilox/kilox_ws/sc";
    int floor = 1;
    load_pcd_and_make_context(map_dir, floor, save_sc_dir);

    ros::spin();

    return 0;
}