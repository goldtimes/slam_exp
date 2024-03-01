#pragma once
#include "edge.hh"
#include "utils/eigen_types.hh"
#include "vertex.hh"

#include <map>
#include <memory>
#include <unordered_map>

namespace vio::backend {
class Problem {
   public:
    /**
     * 问题的类型
     * SLAM问题还是通用的问题
     *
     * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType { SLAM_PROBLEM, GENERIC_PROBLEM };
    using ulong = unsigned long;
    // map是红黑树实现,有序的, unordered_map 是无序的
    using VertexMap = std::map<ulong, std::shared_ptr<Vertex>>;
    using EdgeMap = std::unordered_map<ulong, std::shared_ptr<Edge>>;
    // map unordered_map都不能保存相同的键,但是unordered_multimap可以,也是无序的
    using VertexIdToEdgeMap = std::unordered_multimap<ulong, std::shared_ptr<Edge>>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problem_type);
    ~Problem();
    // 顶点
    bool AddVertex(std::shared_ptr<Vertex> vertex);
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);
    // 边
    bool AddEdge(std::shared_ptr<Edge> edge);
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * @brief 优化过程中被判定为outlier部分的边,前端去除outlier
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

    // 求解问题,传入迭代的次数
    bool Solve(int iterations);
    /**
     * @brief 边缘化一个frame和以它为host的landmark;
     */
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);

    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

    void TestComputePrior();

   private:
    /// Solve的实现，解通用问题
    bool SolveGenericProblem(int iterations);

    /// Solve的实现，解SLAM问题
    bool SolveSLAMProblem(int iterations);

    /// 设置各顶点的ordering_index
    void SetOrdering();

    /// set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /// 构造大H矩阵
    void MakeHessian();

    /// schur求解SBA 稀疏的BA问题
    void SchurSBA();

    /// 解线性方程
    void SolveLinearSystem();

    /// 更新状态变量
    void UpdateStates();

    void RollbackStates();  // 有时候 update 后残差会变大，需要退回去，重来

    /// 计算并更新Prior部分
    void ComputePrior();

    /// 判断一个顶点是否为Pose顶点
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /// 判断一个顶点是否为landmark顶点
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /// 在新增顶点后，需要调整几个hessian的大小
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /// 检查ordering是否正确
    bool CheckOrdering();

    void LogoutVectorSize();

    /// 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// 计算LM算法的初始Lambda
    void ComputeLambdaInitLM();

    /// Hessian 对角线加上或者减去  Lambda
    // LM 算法中 (H+lamda*I) * delta_x = b;
    void AddLambdatoHessianLM();

    void RemoveLambdaHessianLM();

    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();

    /// PCG 迭代线性求解器
    VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter);

   private:
    ProblemType problem_type_;

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;  // LM 迭代退出阈值条件
    double ni_;               // 控制lambda缩放大小

    // LM迭代求解的参数

    MatXX Hessian_;  // 由残差构建出来的海森矩阵
    VecX b_;
    VecX delta_x_;  // 需要求解的增量

    /// 先验部分信息, 曲线拟合中不涉及
    MatXX H_prior_;
    VecX b_prior_;
    MatXX Jt_prior_inv_;
    VecX err_prior_;

    /// SBA的Pose部分
    MatXX H_pp_schur_;
    VecX b_pp_schur_;

    // Heesian 的 Landmark 和 pose 部分
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    // 所有的顶点
    VertexMap verticies_;
    // 所有的边
    EdgeMap edges_;
    // 由vertex id 查询edge
    VertexIdToEdgeMap vertexToEdge_;

    // ordering 相关
    ulong ordering_pose_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;      // 以ordering排序的pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;  // 以ordering排序的landmark顶点

    VertexMap verticies_marginliezd_;

    bool Debug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;
};
}  // namespace vio::backend
