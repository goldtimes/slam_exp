#include "backend/vertex.hh"

namespace vio::backend {
unsigned long global_vertex_id = 0;

Vertex::Vertex(int num_dimension, int local_dimension) {
}
Vertex::~Vertex() {
}

int Vertex::getDimension() const {
    return parameters_.rows();
}

int Vertex::getLocalDimension() const {
    return local_dimension_;
}

void Vertex::plus(const VecX& delta) {
    parameters_ += delta;
}

}  // namespace  vio::backend
