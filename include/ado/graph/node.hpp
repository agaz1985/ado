#ifndef ADO_GRAPH_NODE_HPP
#define ADO_GRAPH_NODE_HPP

#include <stdexcept>

#include "ado/graph/node.h"

namespace ado {
namespace graph {

template <typename T>
Node<T>::Node(const bool requires_grad, const T grad)
    : requires_grad_(requires_grad), grad_(grad), empty_grad_(true) {}

template <typename T>
void Node<T>::backward() {
  this->backward(T({1}, 1));
}

template <typename T>
void Node<T>::backward(const T& grad) {
  if (this->requires_grad()) {
    this->backward_pass(grad);
  }
}

template <typename T>
bool Node<T>::requires_grad() const {
  return this->requires_grad_;
}

template <typename T>
void Node<T>::update_grad(const T& grad) {
  if (!this->requires_grad()) {
    throw std::runtime_error(
        "Backward pass not allowed on tensor that does not require gradient "
        "computation.");
  }
  if (this->empty_grad_ == true) {
    this->grad_ = grad;
    this->empty_grad_ = false;
  } else {
    this->grad_ += grad;
  }
}

template <typename T>
void Node<T>::zero_grad() {
  this->empty_grad_ = true;
}

template <typename T>
T Node<T>::grad() const {
  if (!this->requires_grad()) {
    throw std::runtime_error(
        "Backward pass not allowed on tensor that does not require gradient "
        "computation.");
  }
  return this->grad_;
}

}  // namespace graph
}  // namespace ado

#endif  // ADO_GRAPH_NODE_HPP