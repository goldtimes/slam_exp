#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

// 顶点的定义
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual bool read(std::istream& in) {
    }
    virtual bool write(std::ostream& out) const {
    }

    // 更新
    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector3d(update);
    }

    virtual void setToOriginImpl() override {
        _estimate << 0.0, 0.0, 0.0;
    }
};

// 观测的定义
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CurveFittingEdge(double x) : BaseUnaryEdge(), x_(x) {
    }

    virtual bool read(std::istream& in) {
    }
    virtual bool write(std::ostream& out) const {
    }
    // 残差构建

    virtual void computeError() override {
        const CurveFittingVertex* vertex = static_cast<CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = vertex->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * x_ * x_ + abc(1, 0) * x_ + abc(2, 0));
    }

    // jacobian计算
    virtual void linearizeOplus() override {
        const CurveFittingVertex* vertex = static_cast<CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = vertex->estimate();
        double y = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        _jacobianOplusXi[0] = -x_ * x_ * y;  // de / da
        _jacobianOplusXi[1] = -x_ * y;       // de / db
        _jacobianOplusXi[2] = -y;            // de / dc
    }

   public:
    double x_;
    // y_的值为观测值
};

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    std::default_random_engine rd_engine;
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        double y = std::exp(ae * x * x + be * x + ce) + noise(rd_engine);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    // 构建图优化
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;  // 误差优化项变量维度为3，残差维度为1
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;  // 线性求解器类型

    // 梯度下降的方法
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    // 图模型
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // 图中增加一个顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);
    // 增加边
    for (int i = 0; i < N; ++i) {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        // 设置连接的顶点
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / w_sigma * w_sigma);
        optimizer.addEdge(edge);
    }

    using namespace std;
    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}