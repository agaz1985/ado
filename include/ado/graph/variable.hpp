#ifndef ADO_MATH_TENSOR_HPP
#define ADO_MATH_TENSOR_HPP

#include <stdexcept>

#include "ado/graph/variable.h"
#include "ado/types.h"

namespace ado {
namespace graph {

template <typename T>
Variable<T>::Variable(const Variable<T>& variable)
    : Tensor<T>(variable),
      Node<Tensor<T>>(variable.requires_grad()),
      last_update_(variable->last_update()) {}

template <typename T>
Variable<T>::Variable(const Tensor<T>& tensor, const bool requires_gradient)
    : Tensor<T>(tensor),
      Node<Tensor<T>>(requires_gradient),
      last_update_(xt::zeros_like(tensor)) {}

template <typename T>
Variable<T>::Variable(const Tensor<T>& tensor, const Tensor<T>& grad,
                      const Tensor<T>& last_update)
    : Tensor<T>(tensor),
      Node<Tensor<T>>(true, grad),
      last_update_(last_update) {}

template <typename T>
Variable<T>::Variable(const TensorShape<T>& shape, const T value,
                      const bool requires_gradient)
    : Tensor<T>(shape, value),
      Node<Tensor<T>>(requires_gradient),
      last_update_(xt::zeros(shape)) {}

template <typename T>
void Variable<T>::zeros() {
  (*this) = Variable(xt::zeros_like(*this), this->grad(), this->last_update());
}

template <typename T>
void Variable<T>::rand() {
  (*this) =
      Variable(xt::random::randn<T>(this->shape()), this->grad(),
               this->last_update());  // TODO: it's not always the case it needs
                                      // a gradient. Fix this properly.
}

template <typename T>
void Variable<T>::ones() {
  (*this) = Variable(xt::ones_like(*this), this->grad(), this->last_update());
}

template <typename T>
Tensor<T> Variable<T>::forward() {
  return *this;
}

template <typename T>
void Variable<T>::update(const Tensor<T>& value) {
  if (this->shape() != value.shape()) {
    throw std::runtime_error("Value and tensor shapes do not match.");
  }
  this->last_update_ = value;
  (*this) =
      Variable(this->forward() - value, this->grad(), this->last_update()); // TODO: find a way to update without assignment.
}

template <typename T>
Tensor<T> Variable<T>::last_update() const {
  return this->last_update_;
}

template <typename T>
void Variable<T>::backward_pass(const Tensor<T>& grad) {
  this->update_grad(grad);
}

}  // namespace graph
}  // namespace ado

#endif  // ADO_MATH_TENSOR_HPP
