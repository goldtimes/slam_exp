/**
 * 对极几何约束求解单目相机下的帧与帧直接的姿态估计
 * 单目相机因为尺度的不确定性,需要系统初始化,左右移动相机来完成初始化
 */
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1,
                          std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);

void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch> matches, cv::Mat& R, cv::Mat& t);

/**
 * @brief 像素坐标转相机归一化坐标
 * @param p 像素坐标
 * @param k 相机内参
 */
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);

int main(int argc, char** argv) {
    std::string img_path_1 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/1.png";
    std::string img_path_2 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/2.png";

    cv::Mat img_1 = cv::imread(img_path_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(img_path_2, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    std::vector<cv::DMatch> matches;
    // 特征提取,特征匹配
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;

    // 估计两张图像的R,t
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // 验证本质矩阵E
    // t的反对称矩阵
    // clang-format off
    cv::Mat t_x = (cv::Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1,0),
                                        t.at<double>(2,0), 0, -t.at<double>(0,0),
                                        -t.at<double>(1,0), t.at<double>(0,0), 0);
    // clang-format on
    std::cout << "t^R=\n" << t_x * R << std::endl;

    // 验证对极约束
    // 相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.0, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    for (cv::DMatch m : matches) {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;  // 7.8的公式
        std::cout << "epipoloar constraint=" << d << std::endl;
    }
    return 0;
}

/**
 * @brief 像素坐标转相机归一化坐标, 5.5的公式
 * @param p 像素坐标
 * @param k 相机内参
 *
 */
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
    // clang-format off
    cv::Point2d p_in_camera(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
    // clang-format on
    return p_in_camera;
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch> matches, cv::Mat& R, cv::Mat& t) {
    // 相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 将matches 转换成vector的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (int i = 0; i < (int)matches.size(); ++i) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵F
    cv::Mat fundamental_matrix;
    // 用8个匹配的点去计算F矩阵
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "fundamental_matrix: \n" << fundamental_matrix << std::endl;

    // 计算本质矩阵
    cv::Point2d principal_point(325.1, 249.7);  // c_x, cy 光心标定值
    double focal_lenth = 521;                   //焦距
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_lenth, principal_point);
    std::cout << "essential_matrix is: \n" << essential_matrix << std::endl;

    // 计算单应矩阵,纯旋转的过程有用
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homography_matrix:\n" << homography_matrix << std::endl;

    // 从本质矩阵中计算t,R;
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_lenth, principal_point);
    std::cout << "R is:\n" << R << std::endl;
    std::cout << "t is:\n" << t << std::endl;
}

/**
 * @brief 和上节学的orb_cv一样
 */
void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1,
                          std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches) {
    using namespace cv;
    using namespace std;
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