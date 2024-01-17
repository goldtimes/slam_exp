#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;

float signf(float value) {
    return (value > 0) - (value < 0);
}

void createAlphaMat(cv::Mat &mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            cv::Vec4b &rgba = mat.at<cv::Vec4b>(i, j);
            rgba[0] = 0;
            rgba[1] = 0;
            rgba[2] = 0;
            rgba[3] = 0;  // 0-255 0透明 255不透明
        }
    }
}

Eigen::Matrix4d getExternalParam(PointCloud::Ptr pc_map) {
    float max_x = -10000000000, min_x = 10000000000, max_y = -10000000000, min_y = 10000000000;
    float max_z = -10000000000, min_z = 10000000000;

    float split_size = 50;

    float center_x, center_y, center_z;
    for (int i = 0; i < pc_map->points.size(); i++) {
        PointType pt = pc_map->points[i];

        if (pt.x != pt.x || pt.y != pt.y || pt.z != pt.z) {
            continue;
        }

        // 其他所有点, 包括预先地图中没有定义点类型的点也算进去
        max_x = std::max(pt.x, max_x);
        max_y = std::max(pt.y, max_y);
        min_x = std::min(pt.x, min_x);
        min_y = std::min(pt.y, min_y);
        max_z = std::max(pt.z, max_z);
        min_z = std::min(pt.z, min_z);
    }
    center_x = (max_x + min_x) / 2.0;
    center_y = (max_y + min_y) / 2.0;
    center_z = (max_z + min_z) / 2.0;

    int col = (int)(max_x / split_size) + signf(max_x) * 1;
    int row = (int)(max_y / split_size) + signf(max_y) * 1;

    // center_x center_y 采用图元的正中心
    center_x = col * split_size - signf(max_x) * 25;
    center_y = row * split_size - signf(max_y) * 25;

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ())) *
                           Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())) *
                           Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()));

    Eigen::Vector3d t_;
    t_[0] = (double)center_x;
    t_[1] = (double)center_y;
    t_[2] = (double)0;

    Eigen::Matrix4d ep;
    ep.block(0, 0, 3, 3) = q.matrix();
    ep.block(0, 3, 3, 1) = t_;
    return ep;
}

// 传入当前的scan
void save_meta_color_png_yaml(const std::string &file_path, PointCloud::Ptr &pc_map) {
    // 地图的大小50m
    double map_resolution = 50;
    double half_resolution = map_resolution / 2.0;
    // 1m = 100pixel
    double m_per_pixel = 100;
    float max_x = -100000000000000, min_x = 100000000000000, max_y = -100000000000000, min_y = 100000000000000,
          max_z = -100000000000000, min_z = 100000000000000;
    float min_intensity = 1000;
    float max_intensity = -1000;
    float average_intensity = 0;
    float average_num = 0;

    if (pc_map->points.size() < 20) {
        return;
    }

    Eigen::Matrix4d ep = getExternalParam(pc_map);

    for (int i = 0; i < pc_map->points.size(); i++) {
        PointType pt = pc_map->points[i];
        if (pt.x != pt.x || pt.y != pt.y || pt.z != pt.z) {
            continue;
        }

        // 其他所有点, 包括预先地图中没有定义点类型的点也算进去

        max_x = std::max(pt.x, max_x);
        max_y = std::max(pt.y, max_y);
        min_x = std::min(pt.x, min_x);
        min_y = std::min(pt.y, min_y);
        max_z = std::max(pt.z, max_z);
        min_z = std::min(pt.z, min_z);

        average_num++;
    }

    std::cout << "-----> source max_x:" << max_x << " source max_y:" << max_y << " source min_x:" << min_x
              << " source min_y:" << min_y << " resolution:" << half_resolution << std::endl;
    // auto t_ = ep.block<3, 1>(0, 3);
    // max_x = t_[0] + half_resolution;
    // min_x = t_[0] - half_resolution;

    // max_y = t_[1] + half_resolution;
    // min_y = t_[1] - half_resolution;

    std::cout << "max_x:" << max_x << ",max_y:" << max_y << " min_x:" << min_x << " min_y:" << min_y << std::endl;

    // 2. 得到分辨率
    int border_size = 120;

    // 设置scale表示先在大的地图上描点
    int scale = 2;

    // 设置 1m = 100像素
    int target_width = 1000, target_height = 1000;
    int height = target_height * scale;
    int width = target_width * scale;
    // 获得相对小一些的地图,防止点云太过于接近边界
    // 一个像素代表多少宽度 这个值具体是多少大一点和小一点只要在元文件当中给出则不会有任何问题
    double resolution_x = (max_x - min_x) / (height - 2 * border_size);
    double resolution_y = (max_y - min_y) / (width - 2 * border_size);

    float resolution = 1.0 / m_per_pixel;
    float resolution_inv = m_per_pixel;

    width = std::ceil((max_x - min_x) * resolution_inv);
    height = std::ceil((max_y - min_y) * resolution_inv);
    target_width = width;
    target_height = height;

    std::cout << "height:" << height << ",width:" << width << std::endl;

    cv::Mat mat_pointcloud = cv::Mat(height, width, CV_8UC4, cv::Scalar::all(255));
    cv::Mat mat_pointcloud_old = cv::Mat(height, width, CV_8UC4, cv::Scalar::all(255));
    // exit(0);
    createAlphaMat(mat_pointcloud);
    createAlphaMat(mat_pointcloud_old);

    cv::Mat mat_bg, mat_bg_ori;

    mat_bg = cv::Mat(height, width, CV_8UC4, cv::Scalar::all(0));

    createAlphaMat(mat_bg);
    mat_bg.copyTo(mat_bg_ori);

    cv::Mat mat_bg_resize, mat_bg_resize_old;
    cv::resize(mat_bg, mat_bg_resize, cv::Size(width, height));

    // 定义像素与世界坐标系点的映射关系
    // - 像素中心点的世界坐标为(center_x,center_y)
    // - 像素的col递增方向为世界坐标的x方向
    // - 像素的row递减方向为世界坐标的y方向
    double center_x = (max_x + min_x) / 2.0;
    double center_y = (max_y + min_y) / 2.0;
    double center_z = (max_z + min_z) / 2.0;

    std::cout << "center x:" << center_x << " center y:" << center_y << std::endl;
    // exit(0);
    // 需要首先涂上rigid的颜色,然后将ground覆盖上一层白色
    // 按图元切开 排开 并且按照图元的显示方式进行显示 显示图的拓扑结构
    // 按Z轴
    std::sort(pc_map->points.begin(), pc_map->points.end(), [](const auto &a, const auto &b) { return a.z > b.z; });
    for (int i = 0; i < pc_map->points.size(); i++) {
        PointType pt = pc_map->points[i];

        // 给出左上角的坐标
        float col = width / 2.0 + (pt.x - center_x) / resolution;
        float row = height / 2.0 - (pt.y - center_y) / resolution;

        pcl::PointXYZI pt_rgb;
        pt_rgb.x = col;
        pt_rgb.y = row;
        pt_rgb.z = 0;

        // pt_rgb.intensity = pt.intensity;

        cv::Vec4b color;
        cv::Vec3b color_old;
        double ratio;
        // if (pt.z < 0.0 || pt.z > 1.5) {
        //     continue;
        // }
        ratio = (pt.z - min_z) / (max_z - min_z);
        if (!(col >= 0 && col < width && row >= 0 && row < height)) {
            std::cout << "col:" << col << " row:" << row << " height:" << height << " width:" << width
                      << " center_x:" << center_x << " center_y:" << center_y << " resolution:" << resolution
                      << " pt.x:" << pt.x << " pt.y:" << pt.y << std::endl;
            // LOG_INFO("col:{},row:{} ,height:{},width:{}", col, row, height, width);
            continue;
        }
        // ratio = 1;
        // ratio = (pt.x - min_x) / (max_x - min_x);

        uchar b_old = 0x50 + (0xaf - 0x50) * ratio;
        uchar g_old = 0xbe + (0x55 - 0xbe) * ratio;
        color_old = cv::Vec3b(b_old, g_old, 0);

        color = cv::Vec4b(b_old, g_old, 0, 255);
        // mat_pointcloud.at
        cv::circle(mat_pointcloud, cv::Point2f(col, row), 2, color, cv::FILLED);
    }
    // LOG_INFO("LINE:", __LINE__);
    std::vector<int> indices;
    std::vector<float> distances;

    cv::Mat resize_mat;

    target_width = 5000;
    target_height = 5000;

    float target_x = mat_pointcloud.cols;
    float target_y = mat_pointcloud.rows;
    float scale_pixel = 1;
    while (target_x > target_width && target_y > target_height) {
        scale_pixel *= 2;
        target_x /= 2.0;
        target_y /= 2.0;
    }

    // LOG_INFO("==============================================================");
    // LOG_INFO("target_width:{},target_height:{} ,target_x:{},target_y:{},scale_pixel:{}", target_width, target_height,
    //          target_x, target_y, scale_pixel);
    // LOG_INFO("==============================================================");
    // resize 之后, 分辨率也要相应的设置
    cv::resize(mat_pointcloud, resize_mat, cv::Size((int)target_x, (int)target_y));
    // 图像的大小缩小了，实际代表的大小也就需要调整
    resolution *= scale_pixel;

    time_t curr_time;
    tm *curr_tm;
    char date_string[100];
    time(&curr_time);
    curr_tm = localtime(&curr_time);

    strftime(date_string, 50, "%Y_%m_%d", curr_tm);
    // 添加地图信息

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(20);

    cv::imwrite(file_path + ".jpg", resize_mat, compression_params);
    std::cout << "save image success" << std::endl;

    // center_x = (max_x - min_x) / 2.0;
    // center_y = (max_y - min_y) / 2.0;

    Eigen::Matrix4d inverseEP = ep.inverse();
    Eigen::Vector3d c_t;
    c_t[0] = center_x;
    c_t[1] = center_y;
    c_t[2] = center_z;
    Eigen::Vector3d center_new = inverseEP.block<3, 3>(0, 0) * c_t + inverseEP.block<3, 1>(0, 3);
    // 所有的点都是基于中间的点的外参转换得到的，所以其实转换后的原点origin就是(0,0,0)
    center_x = center_new[0];
    center_y = center_new[1];

    // 保存左上角的坐标
    std::vector<float> origin = {min_x, min_y};
    std::vector<float> max_origin = {max_x, max_y};
    std::vector<float> center_origin = {(float)center_x, (float)center_y};
}

void GenerateBEVImage(PointCloud::Ptr cloud) {
    // 实现将点云转换为鸟瞰图的代码
    // 计算点云边界
    auto minmax_x = std::minmax_element(cloud->begin(), cloud->end(),
                                        [](const PointType &a, const PointType &b) { return a.x < b.x; });
    auto minmax_y = std::minmax_element(cloud->begin(), cloud->end(),
                                        [](const PointType &a, const PointType &b) { return a.y < b.y; });
    double min_x = minmax_x.first->x;
    double max_x = minmax_x.second->x;
    double min_y = minmax_y.first->y;
    double max_y = minmax_y.second->y;

    std::cout << "x: " << min_x << " - " << max_x << std::endl;
    std::cout << "y: " << min_y << " - " << max_y << std::endl;

    const double res = 0.01;
    int image_rows = (int)((max_y - min_y) / res);
    int image_cols = (int)((max_x - min_x) / res);

    float x_center = 0.5 * (max_x + min_x);
    float y_center = 0.5 * (max_y + min_y);

    int image_center_y = image_rows / 2;
    int image_center_x = image_cols / 2;

    cv::Mat bird_image(image_rows, image_cols, CV_8UC4, cv::Scalar(255, 255, 255));
    createAlphaMat(bird_image);

    for (const auto &p : cloud->points) {
        int y = (int)((p.y - y_center) / res + image_center_y);
        int x = (int)((p.x - x_center) / res + image_center_x);
        if (x < 0 || x >= image_cols || y < 0 || y >= image_rows || p.z > 2.5) {
            continue;
        }
        uchar b_old = 0x50 + (0xaf - 0x50);
        uchar g_old = 0xbe + (0x55 - 0xbe);
        // color_old = cv::Vec3b(b_old, g_old, 0);

        auto color = cv::Vec4b(b_old, g_old, 0, 255);
        // mat_pointcloud.at
        cv::circle(bird_image, cv::Point2f(x, y), 2, color, cv::FILLED);
        // bird_image.at<cv::Vec3b>(y, x) = cv::Vec3b(227, 143, 79);
    }

    cv::imwrite("/home/kilox/kilox_ws/global.jpg", bird_image);
    std::cout << "Done" << std::endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pcd_to_bird");
    ros::NodeHandle nh;

    PointCloud::Ptr submap;
    PointCloud::Ptr global_map;
    PointCloud::Ptr filter_global_map;
    submap.reset(new PointCloud);
    global_map.reset(new PointCloud);
    filter_global_map.reset(new PointCloud);
    pcl::io::loadPCDFile("/home/kilox/kilox_ws/submap.pcd", *submap);
    pcl::io::loadPCDFile("/home/kilox/kilox_ws/shinei.pcd", *global_map);
    pcl::VoxelGrid<PointType> filter;
    filter.setLeafSize(0.2, 0.2, 0.2);
    filter.setInputCloud(global_map);
    filter.filter(*filter_global_map);
    pcl::io::savePCDFileASCII("/home/kilox/kilox_ws/shinei_filter.pcd", *filter_global_map);
    save_meta_color_png_yaml("/home/kilox/kilox_ws/submap", submap);
    save_meta_color_png_yaml("/home/kilox/kilox_ws/shinei", global_map);
    // GenerateBEVImage(global_map);
    // ros::spin();
    return 0;
}
