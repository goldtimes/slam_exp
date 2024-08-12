#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <teaser/geometry.h>
#include <teaser/matcher.h>
#include <teaser/registration.h>
#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;
using VoxelGrid = pcl::VoxelGrid<PointType>;
using Normals = pcl::PointCloud<pcl::Normal>;
using FpfhFeature = pcl::PointCloud<pcl::FPFHSignature33>;

PointCloud::Ptr src;
PointCloud::Ptr target;
VoxelGrid filter;

void pcd_to_teaser(const PointCloud::Ptr& cloud_in, teaser::PointCloud& out_cloud) {
    for (size_t i = 0; i < cloud_in->size(); ++i) {
        out_cloud.push_back({cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z});
    }
}

void teaser_to_correspondence(std::vector<std::pair<int, int>>& input, pcl::Correspondences& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(pcl::Correspondence(input[i].first, input[i].second, 0.0));
    }
}

void voxel_filter(const PointCloud::Ptr& cloud_int, PointCloud::Ptr& cloud_out, double voxel_size) {
    filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    filter.setInputCloud(cloud_int);
    filter.filter(*cloud_out);
}

void compute_normals(PointCloud::Ptr& cloud, Normals::Ptr& normals) {
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
    normal_est.setInputCloud(cloud);
    normal_est.setNumberOfThreads(8);
    normal_est.setSearchMethod(kdtree);
    normal_est.setKSearch(10);
    normal_est.compute(*normals);
}

FpfhFeature::Ptr compute_fpfh_feature(PointCloud::Ptr& cloud, Normals::Ptr& normals) {
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    FpfhFeature::Ptr fpfh(new FpfhFeature());
    pcl::FPFHEstimationOMP<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setNumberOfThreads(8);
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchMethod(kdtree);
    fpfh_est.setKSearch(10);
    fpfh_est.compute(*fpfh);
    return fpfh;
}

teaser::RegistrationSolution teaser_registration(PointCloud::Ptr& source, PointCloud::Ptr& target,
                                                 pcl::Correspondences& cur_corr) {
    teaser::PointCloud src_teaser;
    teaser::PointCloud tar_teaser;
    pcd_to_teaser(source, src_teaser);
    pcd_to_teaser(target, tar_teaser);
    // 计算法向量
    Normals::Ptr src_norms(new Normals());
    Normals::Ptr tar_norms(new Normals());
    compute_normals(source, src_norms);
    compute_normals(target, tar_norms);
    // 计算fpfh特征描述子
    FpfhFeature::Ptr src_fpfh = compute_fpfh_feature(source, src_norms);
    FpfhFeature::Ptr tar_fpfh = compute_fpfh_feature(target, tar_norms);
    std::cout << "source descriptors size:" << src_fpfh->size() << ", target descriptors size:" << tar_fpfh->size()
              << std::endl;
    teaser::Matcher matcher;
    auto correspondence =
        matcher.calculateCorrespondences(src_teaser, tar_teaser, *src_fpfh, *tar_fpfh, false, true, false, 0.9);
    std::cout << "correspondence size: " << correspondence.size() << std::endl;
    teaser_to_correspondence(correspondence, cur_corr);

    // 配准
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.05;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    // solver
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src_teaser, tar_teaser, correspondence);
    teaser::RegistrationSolution solution = solver.getSolution();
    return solution;
}

int main(int argc, char** argv) {
    src.reset(new PointCloud());
    target.reset(new PointCloud());

    std::string global_map_path = "/home/kilox/kilox_ws/icp/1/map.pcd";
    std::string src_cloud_path = "/home/kilox/kilox_ws/icp/1/1_scan.pcd";

    pcl::io::loadPCDFile(global_map_path, *target);
    pcl::io::loadPCDFile(src_cloud_path, *src);
    // 体素滤波
    // voxel_filter(src, src, 4);
    // voxel_filter(target, target, 4);
    // 构造旋转矩阵和平移矩阵
    // Eigen::Matrix3d rotation = Eigen::AngleAxisd(M_PI / 3, Eigen::Vector3d::UnitZ()) *
    //                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
    //                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d trans(0.0, 0.0, 0.0);

    Eigen::Matrix4d tf_matrix = Eigen::Matrix4d::Identity();
    tf_matrix.block<3, 3>(0, 0) = rotation;
    tf_matrix.block<3, 1>(0, 3) = trans;

    PointCloud::Ptr src_trans(new PointCloud());
    pcl::transformPointCloud(*src, *src_trans, tf_matrix.cast<float>());

    pcl::Correspondences corrs;
    teaser::RegistrationSolution solution = teaser_registration(src_trans, target, corrs);
    // Compare results
    std::cout << "=====================================" << std::endl;
    std::cout << " TEASER++ Results " << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Expected scale: " << std::endl;
    std::cout << solution.scale << std::endl;
    std::cout << "Expected rotation: " << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "Estimated rotation: " << std::endl;
    std::cout << solution.rotation << std::endl;
    std::cout << std::endl;
    std::cout << "Expected translation: " << std::endl;
    std::cout << trans << std::endl;
    std::cout << "Estimated translation: " << std::endl;
    std::cout << solution.translation << std::endl;
    std::cout << std::endl;

    // 执行变换
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = solution.rotation;
    transformation.block<3, 1>(0, 3) = solution.translation;
    PointCloud::Ptr result_cloud(new PointCloud());

    pcl::transformPointCloud(*src_trans, *result_cloud, transformation);

    // computeRMSE(TCloud, TransformCloud);

    pcl::visualization::PCLVisualizer viewer("Alignment - Teaser");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 255, 0, 0);
    viewer.addPointCloud(target, target_color, "TCloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> reg_color(src, 0, 255, 0);
    viewer.addPointCloud(result_cloud, reg_color, "RegCloud");
    // 对应关系可视化
    // viewer.setWindowName("基于特征描述子的对应");
    // viewer.addCorrespondences<pcl::PointXYZ>(result_cloud, target, corrs, "correspondence");
    // viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "correspondence");

    viewer.spin();

    return 0;
}
