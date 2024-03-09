#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
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

void BAG2o(const VecVector3d &points_3d, VecVector2d &points_2d, const Mat &K, Sophus::SE3 &pose);

int main(int argc, char **argv) {
    std::string img_path_1 = "/home/hang/robot_algo_ws/src/slam_exp/img_data/1.png";
    std::string img_path_2 = "/home/hang/robot_algo_ws/src/slam_exp/img_data/2.png";
    std::string img_depth_path = "/home/hang/robot_algo_ws/src/slam_exp/img_data/1_depth.png";

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

    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3 pose_g2o;
    t1 = chrono::steady_clock::now();
    BAG2o(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
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

/**
 * @brief RGBD相机，有深度信息
 */
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
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        cost = 0.0;
        // 计算cost
        for (int i = 0; i < points_3d.size(); ++i) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            // 投影点
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            Eigen::Vector2d e = points_2d[i] - proj;  // 误差
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;  // jacobian矩阵 7.46公式
            // clang-format off
            J << -fx * inv_z, 0, fx * pc[0] *inv_z2, fx * pc[0] * pc[1] * inv_z2, -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z,
                0, -fy * inv_z , fy * pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2, -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;
            // clang-format on

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        Eigen::Matrix<double, 6, 1> dx;
        dx = H.ldlt().solve(b);
        if (std::isnan(dx[0])) {
            cout << "result is nan" << endl;
            break;
        }
        // 误差没有减小
        if (iter > 0 && cost >= last_cost) {
            cout << "cost: " << cost << ", last cost: " << last_cost << endl;
            break;
        }
        pose = Sophus::SE3::exp(dx) * pose;
        last_cost = cost;
        cout << "iteration " << iter << " cost=" << setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n" << pose.matrix() << endl;
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // 定义顶点的值
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3();
    }
    // 定义顶点的更新方式 x + dx;
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {
    }

    virtual bool write(ostream &out) const override {
    }
};

// 边的定义，像素的残差维度为x,y所以是2
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos3d, const Eigen::Matrix3d &K) : _pos3d(pos3d), _K(K) {
    }

    virtual bool read(istream &in) override {
    }

    virtual bool write(ostream &out) const override {
    }

    // 误差的定义
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3 T = v->estimate();
        // 转换到相机坐标系下
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];  // 归一化
        _error = _measurement - pos_pixel.head<2>();
    }
    // Jacobian的求解
    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3 T = v->estimate();
        // 转换到相机坐标系下
        Eigen::Vector3d pose_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pose_cam[0];
        double Y = pose_cam[1];
        double Z = pose_cam[2];
        double Z2 = Z * Z;
        // clang-format off
        _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
                            0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
        // clang-format on
    }

   public:
    // 世界坐标系下的3D点
    Eigen::Vector3d _pos3d;
    // 相机内参
    Eigen::Matrix3d _K;
};

void BAG2o(const VecVector3d &points_3d, VecVector2d &points_2d, const Mat &K, Sophus::SE3 &pose) {
    // 构建图优化
    // 设定g2o
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;  // pose 维度为6, 像素维度3
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    // 梯度下降的方法可以选择gn, lm, dogleg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // 图模型，稀疏
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    // 创建顶点
    VertexPose *vertex = new VertexPose();
    vertex->setId(0);
    vertex->setEstimate(Sophus::SE3());
    // 添加顶点
    optimizer.addVertex(vertex);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2), K.at<double>(1, 0), K.at<double>(1, 1),
        K.at<double>(1, 2), K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // 边
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    // 求解
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex->estimate().matrix() << endl;
    pose = vertex->estimate();
}