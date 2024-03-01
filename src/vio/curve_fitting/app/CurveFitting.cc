#include <iostream>
#include <random>
#include "vio/curve_fitting/backend/problem.hh"

using namespace vio::backend;

// 曲线方程 ax^2 + bx + c = y; g2o方式的优化构造

/**
 * @brief 曲线模型的顶点
 * 模板参数: 优化变量的维度和数据类型
 */
class CurveFittingVertex : public Vertex {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CurveFittingVertex() : Vertex(3) {
        // abc 三个参数,所以是3维的
    }
    virtual std::string TypeInfo() const {
        return "abc";
    }
};

/**
 * @brief 误差模型
 *  模板参数: 观测值维度,类型, 连接顶点类型
 */
class CurveFittingEdge : public Edge {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CurveFittingEdge(double x, double y) : Edge(1, 1, std::vector<std::string>{"abc"}) {
        // 曲线拟合的过程,残差的维度显然是1  观测的y - 预测的y值
        // 只有一个顶点
        x_ = x;
        y_ = y;
    }

    virtual void ComputeResidual() override {
        // 带优化的参数 abc
        Vec3 abc = verticies_[0]->getParams();
        // 残差的构建
        residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;
    }
    /**
     * @brief 计算残差对abc变量的jacobian 应该是一个 1* 3的矩阵
     */
    virtual void ComputeJacobians() override {
        Vec3 abc = verticies_[0]->getParams();
        double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        Eigen::Matrix<double, 1, 3> jacobian;
        // 幂函数的求导, 先对a,在对b,在对c求导
        jacobian << exp_y * x_ * x_, exp_y * x_, exp_y * 1;
        jacobians_[0] = jacobian;
    }

    virtual std::string TypeInfo() const override {
        return "CurveFittingEdge";
    }

   public:
    double x_;
    double y_;  // 观测值
};

int main(int argc, char** argv) {
    // 构造真实曲线参数
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100;           // 100个数据
    double w_sigma = 1.0;  // 噪声

    std::default_random_engine rd_engine;
    std::normal_distribution<double> noise(0, w_sigma);

    // 构建problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);

    // 构造顶点
    std::shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());
    // 设定带估计参数的初始值
    vertex->setParams(Eigen::Vector3d(0.0, 0.0, 0.0));

    problem.AddVertex(vertex);

    // 构造100次观测,在x处观测到y的值
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        double n = noise(rd_engine);
        // 构造观测的y,并在y上给上噪声
        double y = std::exp(a * x * x + b * x + c) + n;
        // 构造边,每个观测对应的残差函数
        std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        // 设置顶点,类似于图优化中的一元边
        edge->setVertices(edge_vertex);
        // 添加问题到problem中
        problem.AddEdge(edge);
    }
    std::cout << "\nTest CurveFitting start...." << std::endl;
    problem.Solve(30);
    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->getParams().transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;
    return 0;
}