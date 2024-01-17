#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>  //accumulate
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * A  = USV U,V为正交矩阵,S为奇异值矩阵,对角线元素是奇异值, 由大到小的排列
 *
 * @brief 平面拟合, 本质上是超定方程的解法: 特征值解法, svd求解
 * @param points 传入的待拟合的平面啊
 * @param esp 判断平面拟合的阈值
 * @param plane_coeff 拟合平面后的平面参数
 * 这里一定要用模板
 */
template <typename S>
bool FitPlane(std::vector<Eigen::Matrix<S, 3, 1>>& points, Eigen::Matrix<S, 4, 1>& plane_coeffs,
              const double eps = 1e-2) {
    // 平面点小于3,拟合失败
    if (points.size() < 3) return false;
    // 构造超定方程
    Eigen::MatrixXd A(points.size(), 4);
    // Ax+By+Cz = D, 理论上构造5个方程
    // ax+by+cz = 1; // 去除D
    // A_5x4的矩阵 如何求解出来平面参数呢? svd分解
    // mm * mn * nn
    for (int i = 0; i < points.size(); ++i) {
        // 对每行的前三个赋值
        A.row(i).head<3>() = points[i].transpose();
        A.row(i)[3] = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd_full(A, Eigen::ComputeFullU || Eigen::ComputeFullV);
    // 平面拟合中不应该用ComputeFullU
    // std::cout << "u matrix: " << svd_full.computeU() << std::endl;
    // std::cout << "singular matrix: " << svd_full.singularValues().transpose() << std::endl;
    // std::cout << "v matrix: " << svd_full.matrixV() << std::endl;

    std::cout << "u matrix: " << svd.computeU() << std::endl;
    std::cout << "singular matrix: " << svd.singularValues().transpose() << std::endl;
    std::cout << "v matrix: " << svd.matrixV() << std::endl;

    plane_coeffs = svd.matrixV().col(3);

    // check error
    for (int i = 0; i < points.size(); i++) {
        // 点到面的距离
        double err = plane_coeffs.template head<3>().dot(points[i]) + plane_coeffs[3];
        if (err * err > eps) {
            std::cout << "err: " << err << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief 直线拟合
 * @param datas 待拟合的点
 * @param origin 直线参数中的原点
 * @param dir 直线参数中的方向
 * x = dt+p 直线方程
 */
template <typename S>
bool FitLine(std::vector<Eigen::Matrix<S, 3, 1>>& datas, Eigen::Matrix<S, 3, 1>& origin, Eigen::Matrix<S, 3, 1>& dir,
             double eps = 0.2) {
    if (datas.size() < 2) return false;

    // 求平均值

    origin = std::accumulate(datas.begin(), datas.end(), Eigen::Matrix<S, 3, 1>::Zero().eval()) / datas.size();

    // svd 求解
    Eigen::MatrixXd A(datas.size(), 3);
    for (int i = 0; i < datas.size(); ++i) {
        A.row(i) = (datas[i] - origin).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    // A的最大右奇异值
    dir = svd.matrixV().col(0);

    for (const auto& d : datas) {
        if (dir.template cross(d - origin).template squaredNorm() > eps) {
            return false;
        }
    }
    return true;
}

void PlaneFiteTest() {
    // 构造真实的平面参数
    Eigen::Vector4d true_plane_coeffs(0.1, 0.2, 0.3, 0.4);
    // 参数归一化
    true_plane_coeffs.normalize();
    // 生成平面点
    cv::RNG rng;
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 10; ++i) {
        // 随机点
        Eigen::Vector3d p(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
        // 点到平面的距离
        double n4 = -p.dot(true_plane_coeffs.head<3>()) / true_plane_coeffs[3];
        // 构造平面上的点
        p = p / (n4 + std::numeric_limits<double>::min());  // 加上最小值，防止除零
        // 对p进行扰动
        p += Eigen::Vector3d(rng.gaussian(0.01), rng.gaussian(0.01), rng.gaussian(0.01));

        points.push_back(p);
        std::cout << "res of p:" << p.dot(true_plane_coeffs.head<3>()) + true_plane_coeffs[3] << std::endl;
    }
    // 上面产生了一系列的点,在某个平面周边的点集合
    Eigen::Vector4d est_plane_coeffs;
    if (FitPlane(points, est_plane_coeffs)) {
        std::cout << "estimated coeffs: " << est_plane_coeffs.transpose() << ", true:" << true_plane_coeffs.transpose()
                  << std::endl;
    } else {
        std::cout << "plane fitting failed";
    }
}

/**
 * 3d空间中的直线拟合
 */
void LineFiteTest() {
    cv::RNG rng;
    // 定义直线的参数
    Eigen::Vector3d true_line_origin(0.1, 0.2, 0.3);
    Eigen::Vector3d true_line_dir(0.4, 0.5, 0.6);
    true_line_dir.normalize();
    // 生成一定数量直线上的点
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 100; ++i) {
        double t = rng.uniform(-1.0, 1.0);
        Eigen::Vector3d p = true_line_origin + true_line_dir * t;
        p += Eigen::Vector3d(rng.gaussian(0.01), rng.gaussian(0.01), rng.gaussian(0.01));
        points.push_back(p);
    }
    Eigen::Vector3d est_origin;
    Eigen::Vector3d est_dir;
    if (FitLine(points, est_origin, est_dir)) {
        std::cout << "est origin:" << est_origin.transpose() << ",true_origin:" << true_line_origin.transpose()
                  << std::endl;
        std::cout << "est dir:" << est_dir.transpose() << ",true_dir:" << true_line_dir.transpose() << std::endl;
    } else {
        std::cout << "line fitting failed" << std::endl;
    }
}

int main(int argc, char** argv) {
    // 构造平面点
    PlaneFiteTest();
    // 构造直线点
    LineFiteTest();
    return 0;
}