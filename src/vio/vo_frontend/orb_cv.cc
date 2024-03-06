#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
    std::string img_path_1 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/1.png";
    std::string img_path_2 = "/home/kilox/catkin_ws/robot_algo_ws/src/slam_exp/img_data/2.png";

    // cv读取图像
    cv::Mat img_1 = cv::imread(img_path_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(img_path_2, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // 初始化
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    // orb特征检测器
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // 特征匹配器
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-1. 检测oriented Fast角点, 要记住orb特征点提取方法
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-2. 根据特征点的坐标,计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds." << std::endl;

    // 绘制特征点
    cv::Mat out_img1;
    cv::drawKeypoints(img_1, keypoints_1, out_img1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::namedWindow("ORB features", cv::WINDOW_NORMAL);
    cv::imshow("ORB features", out_img1);
    // -3. 描述子匹配,暴力匹配

    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    // 需要清楚如何做描述子的匹配, 计算描述子的距离
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds. " << std::endl;

    // -4. 匹配点筛选
    auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
        return m1.distance < m2.distance;
    });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    std::cout << "max dist:" << max_dist << ", min_dis: " << min_dist << std::endl;

    // 筛选描述子
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; ++i) {
        // 如果大于2倍的最小距离,匹配错误
        // 当最小距离比较小时,我们给以超参
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }
    using namespace cv;
    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::namedWindow("all matches", cv::WINDOW_NORMAL);
    imshow("all matches", img_match);
    cv::namedWindow("good matches", cv::WINDOW_NORMAL);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;
}