#include <sophus/se3.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

using VecVector3d = vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using VecVector2d = vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

/**
 * @brief 手写高斯牛顿BA
 */

void BAGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3 &pose);

int main(int argc, char **argv) {
    std::string img_path_1 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/1.png";
    std::string img_path_2 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/2.png";
    std::string img_depth_path = "";

    Mat img_1 = imread(img_path_1, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(img_path_2, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 深度图
    Mat d1 = imread(img_depth_path, CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    for (DMatch m : matches) {
        // 获取深度信息, 将x,y取整获得像素位置
        ushort d = d1.ptr<unsigned char>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) {
            continue;
        }
        float dd = d / 5000.0;
        // 像素点->camera点
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.emplace_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.emplace_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    cout << "cv solvePnP" << endl;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    // 求r为旋转向量
    Mat R;
    cv::Rodrigues(r, R);  // Rodrigues公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3 pose_gn;
    t1 = chrono::steady_clock::now();
    BAGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void BAGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3 &pose) {
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    const int iterations = 10;
    double cost = 0.0, last_cost = 0.0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    // 
    for (int iter = 0; iter < iterations; ++iter) {
    
    }
}