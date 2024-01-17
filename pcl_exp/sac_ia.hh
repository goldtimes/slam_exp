#pragma once
#include <pcl/io/pcd_io.h>    // 读取点云文件
#include <pcl/point_cloud.h>  // pcl点云的类型
#include <pcl/point_types.h>  // pcl点的类型

// pcl特征
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
// 法向量特征
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
// sac-ia算法
#include <pcl/registration/ia_ransac.h>
// kdtree搜索
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <map>
#include "tic_toc.hh"

using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;
// 法向量信息
using SurfaceNormals = pcl::PointCloud<pcl::Normal>;
// 特征直方图
using FPFHFeatures = pcl::PointCloud<pcl::FPFHSignature33>;
// kdtree搜索
using SearchMethod = pcl::search::KdTree<PointType>;

class FeatureCloud {
   public:
    FeatureCloud() : kd_tree_(new SearchMethod), normal_radius_(1.2f), feature_radius_(1.21f) {
    }
    // 输入点云
    void setInputCloud(const PointCloud::Ptr& cloud) {
        input_cloud_.reset(new PointCloud);
        input_cloud_ = cloud;
        processInputCloud();
    }

    // 获取特征描述符
    FPFHFeatures::Ptr getFPFHFeatures() {
        return fpfh_features_;
    }
    SurfaceNormals::Ptr getSurfaceNormals() {
        return surface_normals_;
    }
    PointCloud::Ptr getPointCloud() {
        return input_cloud_;
    }

   private:
    bool processInputCloud() {
        std::cout << "start feature input cloud" << std::endl;
        if (input_cloud_->empty()) {
            std::cerr << "input cloud is empty" << std::endl;
            return false;
        }
        // // 计算法向量
        timer.tic();
        computeSurfaceNormals();
        auto time_user = timer.toc();
        std::cout << "normals computed in " << time_user << " ms" << std::endl;
        // // 计算特征
        timer.tic();
        computeFPFHFeatures();
        time_user = timer.toc();
        std::cout << "FPFHFeatures computed in " << time_user << " ms" << std::endl;
        return true;
    }

    void computeSurfaceNormals() {
        // 创建法向量点云
        surface_normals_ = SurfaceNormals::Ptr(new SurfaceNormals());
        pcl::NormalEstimation<PointType, pcl::Normal> normal_est;
        // 设置输入点云
        normal_est.setInputCloud(input_cloud_);
        // 设置kdtree搜索方法
        normal_est.setSearchMethod(kd_tree_);
        // 设置搜索半径
        normal_est.setRadiusSearch(normal_radius_);
        normal_est.compute(*surface_normals_);
        // 可视化点云显示
    }

    void computeFPFHFeatures() {
        // 创建特征描述子
        fpfh_features_.reset(new FPFHFeatures());
        pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;

        fpfh_est.setInputCloud(input_cloud_);
        fpfh_est.setInputNormals(surface_normals_);
        fpfh_est.setSearchMethod(kd_tree_);
        fpfh_est.setRadiusSearch(feature_radius_);
        fpfh_est.compute(*fpfh_features_);
    }

   private:
    PointCloud::Ptr input_cloud_;
    SurfaceNormals::Ptr surface_normals_;
    FPFHFeatures::Ptr fpfh_features_;
    SearchMethod::Ptr kd_tree_;
    float normal_radius_;
    float feature_radius_;
    TicToc timer;
};

class TemplateAligment {
   public:
    struct MetaInfo {
        PointCloud::Ptr lidar_data;
        int level;
        std::string name;
        Eigen::Matrix4d T;
    };

    struct Result {
        float fitness_score;
        Eigen::Matrix4f final_transformation;
        bool has_converged;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

   public:
    // 构造函数,加载全部的地图点云
    TemplateAligment(const std::string& map_dir, int level) {
        global_map_ptr_.reset(new PointCloud());
        current_scan_ptr_.reset(new PointCloud());
        feature_cloud_ptr_ = std::make_shared<FeatureCloud>();
        traj_cloud.reset(new PointCloud());
        load_pcd(map_dir, level);

        min_sample_distance_ = 0.01f;
        max_correspondence_distance_ = 10.0f;
        max_iterations_ = 5000;
        sac_ia_.setTransformationEpsilon(0.0001);
        sac_ia_.setMinSampleDistance(min_sample_distance_);
        sac_ia_.setMaxCorrespondenceDistance(max_correspondence_distance_);
        sac_ia_.setMaximumIterations(max_iterations_);
    }

    void setTargetCloud(const FeatureCloud& target_cloud) {
        feature_target_cloud_ = target_cloud;
        sac_ia_.setInputTarget(feature_target_cloud_.getPointCloud());
        sac_ia_.setTargetFeatures(feature_target_cloud_.getFPFHFeatures());
    }

    void align(FeatureCloud& source_cloud, std::vector<Result>& results) {
        int level = 1;
        auto metas = meta_group_[level];
        for (int i = 0; i < metas.size(); ++i) {
            std::cout << "aligning..." << std::endl;
            // 生成global和scan的特征直方图
            feature_target_cloud_.setInputCloud(metas[i]->lidar_data);
            // sac_ia 设置目标点云
            setTargetCloud(feature_target_cloud_);
            sac_ia_.setInputCloud(source_cloud.getPointCloud());
            sac_ia_.setSourceFeatures(source_cloud.getFPFHFeatures());

            PointCloud::Ptr out_cloud(new PointCloud());
            sac_ia_.align(*out_cloud);
            pcl::io::savePCDFile("/home/kilox/match_cloud.pcd", *out_cloud);
            Result result;
            result.fitness_score = (float)sac_ia_.getFitnessScore(max_correspondence_distance_);
            result.final_transformation = sac_ia_.getFinalTransformation();
            result.has_converged = sac_ia_.hasConverged();
            results.push_back(result);
        }
    }

    void align_single(FeatureCloud& source_cloud, std::vector<Result>& results) {
        std::cout << "aligning..." << std::endl;
        // 生成global和scan的特征直方图
        feature_target_cloud_.setInputCloud(global_map_ptr_);
        // sac_ia 设置目标点云
        setTargetCloud(feature_target_cloud_);
        sac_ia_.setInputCloud(source_cloud.getPointCloud());
        sac_ia_.setSourceFeatures(source_cloud.getFPFHFeatures());

        PointCloud::Ptr out_cloud(new PointCloud());
        sac_ia_.align(*out_cloud);
        pcl::io::savePCDFile("/home/kilox/match_cloud.pcd", *out_cloud);
        Result result;
        result.fitness_score = (float)sac_ia_.getFitnessScore(max_correspondence_distance_);
        result.final_transformation = sac_ia_.getFinalTransformation();
        result.has_converged = sac_ia_.hasConverged();
        results.push_back(result);
    }

    bool findBestAligment(const PointCloud::Ptr& source_cloud, Eigen::Matrix4d& eigen_matrixd,
                          pcl::registration::TransformationEstimation<PointType, PointType>::Matrix4& T_prev,
                          float& score) {
        std::vector<Result> results;
        feature_source_cloud_.setInputCloud(source_cloud);

        align_single(feature_source_cloud_, results);
        float lowest_score = results[0].fitness_score;
        int count = 0;
        int best_index = 0;
        for (int i = 1; i < results.size(); i++) {
            Result result = results[i];
            std::cout << "result.fitness_score: " << result.fitness_score << "  lowest_score: " << lowest_score
                      << std::endl;
            if (result.fitness_score < lowest_score) {
                lowest_score = result.fitness_score;
                best_index = i;
            }
        }

        auto best_result = results[best_index];
        std::cout << "*********** RANSAC==*******=======结果是否收敛：\n" << best_result.has_converged << std::endl;

        std::cout << "匹配分数score: " << best_result.fitness_score << std::endl;
        std::cout << "变换矩阵：\n" << best_result.final_transformation << std::endl;
        Eigen::Matrix4f transformationMatrix4f = best_result.final_transformation;

        eigen_matrixd = transformationMatrix4f.cast<double>();
        std::cout << "eigen_matrix: " << eigen_matrixd << std::endl;
        return true;
    }

   private:
    void load_pcd(const std::string& map_dir, int level) {
        int pcd_count = 0;
        // 如果存在就直接加载
        boost::filesystem::path global_map_path("/home/kilox/shinei.pcd");
        if (boost::filesystem::exists(global_map_path)) {
            pcl::io::loadPCDFile(global_map_path.string(), *global_map_ptr_);
            std::cout << "load global map pcd file success!" << std::endl;
            return;
        }

        boost::filesystem::path dir(map_dir);
        if (boost::filesystem::exists(dir)) {
            for (const auto& iter : boost::filesystem::directory_iterator(dir)) {
                // 获取图元的目录
                if (boost::filesystem::is_directory(iter)) {
                    std::stringstream ss;
                    ss << iter.path().string() << "/data.yaml";
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
                        if (meta_group_.find(level) == meta_group_.end()) {
                            meta_group_[level] = std::vector<std::shared_ptr<MetaInfo>>();
                        }
                        meta_group_[level].push_back(std::make_shared<MetaInfo>());
                        auto& meta_info = meta_group_[level].back();
                        meta_info->level = level;
                        meta_info->lidar_data.reset(new PointCloud);
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
            pcl::io::savePCDFileBinary("/home/kilox/global_map.pcd", *global_map_ptr_);
        }
    }

   private:
    PointCloud::Ptr global_map_ptr_;    // 全局地图
    PointCloud::Ptr current_scan_ptr_;  // 当前的scan

    std::map<int, std::vector<std::shared_ptr<MetaInfo>>> meta_group_;
    PointCloud::Ptr traj_cloud;
    TicToc timer;
    std::shared_ptr<FeatureCloud> feature_cloud_ptr_;
    // sac_ia
    pcl::SampleConsensusInitialAlignment<PointType, PointType, pcl::FPFHSignature33> sac_ia_;
    float min_sample_distance_;
    float max_correspondence_distance_;
    float max_iterations_;
    FeatureCloud feature_target_cloud_;
    FeatureCloud feature_source_cloud_;
};