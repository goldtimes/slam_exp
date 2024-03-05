#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
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
    // jacobian计算

   public:
    double x_;
    // y_的值为观测值
};