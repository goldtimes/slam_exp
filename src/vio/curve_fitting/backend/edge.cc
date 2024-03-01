#include "vio/curve_fitting/backend/edge.hh"
#include <iostream>
#include "vio/curve_fitting/backend/vertex.hh"

namespace vio::backend {

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies, const std::vector<std::string>& verticies_types) {
    residual_.resize(residual_dimension, 1);
    if (!verticies_types.empty()) {
        verticies_types_ = verticies_types;
    }
    jacobians_.resize(num_verticies);
    id_ = global_edge_id++;
    // 构造函数中设置了信息矩阵
    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;
}

Edge::~Edge() {
}

double Edge::Chi2() {
    return residual_.transpose() * information_ * residual_;
}

bool Edge::checkValid() {
    if (!verticies_types_.empty()) {
        // check type info
        for (size_t i = 0; i < verticies_.size(); ++i) {
            if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
                std::cout << "Vertex type does not match, should be " << verticies_types_[i] << ", but set to "
                          << verticies_[i]->TypeInfo() << std::endl;
                return false;
            }
        }
    }
    return true;
}
}  // namespace vio::backend