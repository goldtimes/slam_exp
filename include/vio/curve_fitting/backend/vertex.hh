#pragma once
#include "utils/eigen_types.hh"

namespace vio::backend {
/**
 * @brief 顶点，对应一个parameter block
 * 变量值以vector存储，构造时需要指定维度
 */
class Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   public:
    /**
     * @brief 构造函数
     * @param num_dimension 顶点自身的维度
     * @param local_dimension 本地参数化维度，为-1的时候认为和顶点本身维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    virtual ~Vertex();

    int getDimension() const;
    int getLocalDimension() const;
    unsigned long getId() const {
        return id_;
    }

    VecX getParams() const {
        return parameters_;
    }

    VecX& getParams() {
        return parameters_;
    }

    void setParams(const VecX& params) {
        parameters_ = params;
    }

    /**
     * @brief 加法
     */
    virtual void plus(const VecX& delta);
    // 纯虚函数
    virtual std::string TypeInfo() const = 0;

    int OrderingId() const {
        return ordering_id_;
    }

    void setOrderingId(unsigned long id) {
        ordering_id_ = id;
    }

    void setFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    bool isFixed() const {
        return fixed_;
    }

   private:
    VecX parameters_;                //实际存储的变量值
    int local_dimension_;            //局部参数化维度
    unsigned long id_;               //  顶点的id，自动生成
    unsigned long ordering_id_ = 0;  // 在problem排行后的id
    bool fixed_ = false;             // 是否固定
};
}  // namespace vio::backend