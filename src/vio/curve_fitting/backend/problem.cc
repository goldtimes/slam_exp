#include "vio/curve_fitting/backend/problem.hh"
#include <glog/logging.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include "utils/tic_toc.hh"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace vio::backend {
void Problem::LogoutVectorSize() {
}

Problem::Problem(ProblemType problem_type) {
    problem_type_ = problem_type;
    LogoutVectorSize();
    verticies_marginliezd_.clear();
}

Problem::~Problem() {
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    // 如果已经添加了顶点
    if (verticies_.find(vertex->getId()) != verticies_.end()) {
        // LOG(INFO) << "vertex has added";
        return false;
    } else {
        verticies_.insert({vertex->getId(), vertex});
    }
    return true;
}

bool Problem::AddEdge(std::shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert({edge->Id(), edge});
    } else {
        return false;
    }
    for (auto& vertex : edge->getVertices()) {
        vertexToEdge_.insert({vertex->getId(), edge});
    }
    return true;
}

bool Problem::Solve(int iterations) {
    if (verticies_.size() == 0 || edges_.size() == 0) {
        // LOG(ERROR) << "Cannot solve problem without edges or verticies";
        return false;
    }
    TicToc t_solve;
    // 统计优化变量的维度,构建H矩阵需要知道维度
    SetOrdering();
    // 遍历edge, 构建J^T * J矩阵
    MakeHessian();
    // LM初始化
    ComputeLambdaInitLM();
    bool stop = false;
    int iter = 0;
    // 没有设置停止标志位或者迭代次数小于iterations
    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << ", chi= " << currentChi_ << ", lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        // 不断尝试 Lambda, 直到成功迭代一步
        while (!oneStepSuccess) {
            // (H+lamda*I) * deltaX = b, 所以这里将lamda放到hessian矩阵中
            AddLambdatoHessianLM();
            // 求解Hx=b方程
            SolveLinearSystem();
            RemoveLambdaHessianLM();
            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                std::cout << "stoped, delta_x  norm: " << delta_x_.squaredNorm() << ",false_cnt:" << false_cnt
                          << std::endl;
                stop = true;
                break;
            }
            // 更新状态量
            UpdateStates();
            // lm 中判断当前步长是否可行已经lambda更新
            oneStepSuccess = IsGoodStepInLM();
            if (oneStepSuccess) {
                MakeHessian();
                false_cnt = 0;
            } else {
                false_cnt++;
                RollbackStates();  // 误差没下降,回滚
            }
        }
        iter++;
        // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
        if (sqrt(currentChi_) <= stopThresholdLM_) {
            std::cout << "chi 下降1e6倍" << std::endl;
            stop = true;
        }
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    return true;
}  // namespace vio::backend

void Problem::SetOrdering() {
    ordering_pose_ = 0;
    ordering_landmarks_ = 0;
    ordering_generic_ = 0;

    for (auto vertex : verticies_) {
        ordering_generic_ += vertex.second->getLocalDimension();
    }
    // LOG(INFO) << "ordering: " << ordering_generic_;
    std::cout << "ordering: " << ordering_generic_ << std::endl;
}

void Problem::MakeHessian() {
    TicToc t_h;
    // 直接构造大的hessin矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));
    // 变量每个edge,计算残差,最后计算H矩阵
    for (auto& edge : edges_) {
        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        auto jacobians = edge.second->getJacobians();
        auto verticies = edge.second->getVertices();
        assert(jacobians.size() == verticies.size());
        // vertices size is 1
        for (size_t i = 0; i < verticies_.size(); ++i) {
            auto v_i = verticies_[i];
            if (v_i->isFixed()) {  // 顶点是固定的,不需要被修改,那么他的jabocian矩阵为0
                continue;
            }
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->getLocalDimension();
            // 3x1的向量和1*1的向量相乘, 信息矩阵的维度=残差的维度
            MatXX JtW = jacobian_i.transpose() * edge.second->getInformation();
            // 遍历i后面的顶点,假设这里是两个顶点
            /**
             *  [ {3x1}(j_1)
             *    {3x1}(j_2) ]_6x1  *  [ {3x1}j_1, {3x1}j_2]1x6
             *  就是简单的矩阵乘法
             */
            for (size_t j = i; j < verticies_.size(); ++j) {
                auto v_j = verticies_[j];
                if (v_j->isFixed()) {
                    continue;
                }
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->getLocalDimension();
                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;
                // 将所有hessin矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
            }
            b.segment(index_i, dim_i).noalias() -= JtW * edge.second->getResidual();
        }
    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

// LM
void Problem::ComputeLambdaInitLM() {
    ni_ = 2;
    currentLambda_ = -1;
    currentChi_ = 0.0;
    // 遍历所有的edge,获取chi2误差
    for (const auto& edge : edges_) {
        currentChi_ += edge.second->Chi2();
    }
    // std::cout << "currentChi: " << currentChi_ << std::endl;
    // 判断是否有先验
    if (err_prior_.rows() > 0) {
        currentChi_ += err_prior_.norm();
    }
    // 设置停止的阈值,取出最大的hessien矩阵值
    stopThresholdLM_ = 1e-6 * currentChi_;
    assert(Hessian_.rows() == Hessian_.cols() && "hessian is not squart");
    double maxDiagonal = 0;
    size_t size = Hessian_.cols();
    for (int i = 0; i < size; ++i) {
        maxDiagonal = std::max(std::fabs(Hessian_(i, i)), maxDiagonal);
    }
    currentLambda_ = 1e-5 * maxDiagonal;
}

void Problem::AddLambdatoHessianLM() {
    size_t size = Hessian_.cols();
    for (int i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::SolveLinearSystem() {
    delta_x_ = Hessian_.inverse() * b_;
}

void Problem::RemoveLambdaHessianLM() {
    size_t size = Hessian_.cols();
    for (int i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

void Problem::UpdateStates() {
    // 遍历所有顶点
    for (auto vertex : verticies_) {
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->getLocalDimension();
        // segment(i,j) 从i开始取j个数据
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->plus(delta);
    }
}

void Problem::RollbackStates() {
    for (auto& vertex : verticies_) {
        int idx = vertex.second->OrderingId();
        int dimension = vertex.second->getLocalDimension();
        Vec3 delta = delta_x_.segment(idx, dimension);
        // std::cout << "vertex: " << vertex.second->getParams().transpose() << std::endl;
        vertex.second->plus(-delta);
        // std::cout << "vertex: " << vertex.second->getParams().transpose() << std::endl;
    }
}

bool Problem::IsGoodStepInLM() {
    double scale = 0;
    scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-3;  // 防止scale为0

    double tempChi = 0.0;
    for (auto edge : edges_) {
        // 更新状态后，重新计算残差
        edge.second->ComputeResidual();
        tempChi += edge.second->Chi2();
    }
    std::cout << "scale:" << scale << std::endl;
    std::cout << "currentChi:" << currentChi_ << ", tempChi" << tempChi << std::endl;
    double rho = (currentChi_ - tempChi) / scale;
    std::cout << "rho:" << rho << std::endl;
    if (rho > 0 && std::isfinite(tempChi)) {
        double alpha = 1.0 - std::pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2.0 / 3.0);
        double scaleFactor = (std::max)(1.0 / 3.0, alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentChi_ *= ni_;
        ni_ *= 2;
        return false;
    }
}

}  // namespace vio::backend