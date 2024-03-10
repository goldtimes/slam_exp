/**
 * LK光流的特征跟踪来估计相机运动
 */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

string file_1 = "/home/hang/robot_algo_ws/src/slam_exp/img_data/LK1.png";  // first image
string file_2 = "/home/hang/robot_algo_ws/src/slam_exp/img_data/LK2.png";  // second image

// 光流的跟踪器
class OpticalFlowTracker {
   public:
    OpticalFlowTracker(const Mat& img1_, const Mat& img2_, const vector<KeyPoint>& kp1_, vector<KeyPoint>& kp2_,
                       vector<bool>& success_, bool inverse_ = true, bool has_initial_ = false)
        : img1(img1_),
          img2(img2_),
          kp1(kp1_),
          kp2(kp2_),
          success(success_),
          inverse(inverse_),
          has_initial(has_initial_) {
    }
    /**
     * @brief 计算一定范围内的光流
     */
    void calculateOpticalFlow(const Range& range);

   private:
    const Mat& img1;
    const Mat& img2;
    const vector<KeyPoint>& kp1;
    vector<KeyPoint>& kp2;
    vector<bool>& success;
    bool inverse = true;
    bool has_initial = false;
};
/**
 * 单层的光流
 */
void OpticalFlowSingleLevel(const Mat& img1, const Mat& img2, const vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                            vector<bool>& success, bool inverse = false, bool has_initial_guess = false);

void OpticalFlowMutilLevel(const Mat& img1, const Mat& img2, const vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                           vector<bool>& success, bool inverse = false);

/**
 * 差值法获得像素的值 the interpolated value of this pixel
 */
inline float GetPixelValue(const Mat& img, float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) x = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1) +
           (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char** argv) {
    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);  // 图片本来就是灰度
    Mat img2 = imread(file_2, 0);

    // 关键点检查
    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);  // maximum 500 keypoints
    detector->detect(img1, kp1);

    // 用单层的光流法来跟踪关键点
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // 多层的光流法来跟踪关键点
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMutilLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // opencv自带的光流法来验证
    vector<Point2f> pt1, pt2;
    for (auto& kp : kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // 单层光流算法 跟踪的特征点
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    // 多层光流算法 跟踪的特征点
    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    // 绘制关键点
    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);
    return 0;
}

void OpticalFlowSingleLevel(const Mat& img1, const Mat& img2, const vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                            vector<bool>& success, bool inverse = false, bool has_initial_guess = false) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial_guess);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range& range) {
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; ++i) {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }
        double cost = 0, lastCost = 0;
        // 根据8.2的公式推导，维度为2
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;

        bool succ = true;  // indicate if this point succeeded
        for (int iter = 0; iter < iterations; ++iter) {
            if (inverse == false) {
                // 反向光流
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;
            // 某个特征点周围的像素来最小二乘
            for (int x = -half_patch_size; x < -half_patch_size; ++i) {
                for (int y = -half_patch_size; y < -half_patch_size; ++i) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                                          GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                                   0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                                          GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                                          GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                                   0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                                          GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }
            }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }
        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}
