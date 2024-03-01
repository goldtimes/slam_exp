#pragma once
#include <memory>
#include <string>
#include "utils/eigen_types.hh"

namespace vio::backend {
/**
 * @brief 负责计算残差(预测-观测)
 * 代价函数= 残差 * 信息矩阵 * 残差，后端求和后最小化
 */

class Vertex;

class Edge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   public:
    /**
     * @brief 构造函数, 会自动分配雅可比空间
     * @param residual_dimension 残差维度
     * @param num_verticies 顶点数量
     * @param verticies_types 顶点类型，可以不给
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string>& verticies_types = std::vector<std::string>());
    ~Edge();

    unsigned long Id() const {
        return id_;
    }

    bool addVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    bool setVertices(const std::vector<std::shared_ptr<Vertex>>& vertices) {
        verticies_ = vertices;
        return true;
    }

    std::shared_ptr<Vertex> getVertex(int id) const {
        return verticies_[id];
    }

    std::vector<std::shared_ptr<Vertex>> getVertices() const {
        return verticies_;
    }

    size_t getNumVertices() const {
        return verticies_.size();
    }

    /// 返回边的类型信息，在子类中实现
    virtual std::string TypeInfo() const = 0;

    /**
     * @brief 计算残差
     */
    virtual void ComputeResidual() = 0;

    /**
     * @brief 计算雅可比矩阵
     */
    virtual void ComputeJacobians() = 0;

    /**
     * @brief 计算平方误差
     */
    double Chi2();

    VecX getResidual() const {
        return residual_;
    }

    std::vector<MatXX> getJacobians() const {
        return jacobians_;
    }

    MatXX getInformation() const {
        return information_;
    }

    void setInformation(const MatXX information) {
        information_ = information;
    }
    // 设置观测
    void setObservation(const VecX& observation) {
        observation_ = observation;
    }

    VecX getObservation() const {
        return observation_;
    }
    bool checkValid();

    int OrderingId() const {
        return ordering_id_;
    }

    void setOrderingId(int id) {
        ordering_id_ = id;
    }

   protected:
    unsigned long id_;                                // 边的id
    int ordering_id_;                                 // 加入problem中排序后的点
    std::vector<std::string> verticies_types_;        // 各类顶点类型信息
    std::vector<std::shared_ptr<Vertex>> verticies_;  // 边的各顶点
    VecX residual_;                                   //残差
    std::vector<MatXX> jacobians_;  //雅可比矩阵，每个雅可比的维度是residual *vertex[i];
    MatXX information_;
    VecX observation_;  // 观测信息
};
}  // namespace vio::backend