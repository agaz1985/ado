#ifndef ADO_MATH_TENSOR_HPP
#define ADO_MATH_TENSOR_HPP

#include <stdexcept>

#include "ado/graph/variable.h"
#include "ado/types.h"

namespace ado {
namespace graph {

template <typename T>
Variable<T>::Variable(const Variable<T>& variable)
    : Tensor<T>(variable), Node<Tensor<T>>(variable.requires_grad()) {}

template <typename T>
Variable<T>::Variable(const Tensor<T>& tensor, const bool requires_gradient)
    : Tensor<T>(tensor), Node<Tensor<T>>(requires_gradient) {}

template <typename T>
Variable<T>::Variable(const TensorShape<T>& shape, const T value,
                      const bool requires_gradient)
    : Tensor<T>(shape, value), Node<Tensor<T>>(requires_gradient) {}

template <typename T>
void Variable<T>::zeros() {
  (*this) = Variable(xt::zeros_like(*this), this->requires_grad());
}

template <typename T>
void Variable<T>::ones() {
  (*this) = Variable(xt::ones_like(*this),
                     this->requires_grad());  // TODO: check this and check if
                                              // we need to copy the grad.
}

template <typename T>
Tensor<T> Variable<T>::forward() {
  return *this;
}

template <typename T>
void Variable<T>::update(const Tensor<T>& value) {
  if (this->shape() != value.shape()) {
    throw std::runtime_error("Gradient and tensor shapes do not match.");
  }

  (*this) = Variable(value, this->requires_grad());
}

template <typename T>
void Variable<T>::backward_pass(const Tensor<T>& grad) {
  this->update_grad(grad);
}

}  // namespace graph
}  // namespace ado

#endif  // ADO_MATH_TENSOR_HPP
