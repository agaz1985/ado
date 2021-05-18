#ifndef ADO_MATH_UNARY_OPERATORS_HPP
#define ADO_MATH_UNARY_OPERATORS_HPP

#include <xtensor/xsort.hpp>

#include "ado/math/unary_operators.h"

namespace ado {
namespace math {

// Exp operator.

template <typename T>
ExpOperator<T>::ExpOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
Tensor<T> ExpOperator<T>::forward() {
  return xt::exp(this->operands_[0]->forward());
}

template <typename T>
void ExpOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(xt::exp(this->operands_[0]->forward()) * grad);
}

// Log operator.

template <typename T>
LogOperator<T>::LogOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
Tensor<T> LogOperator<T>::forward() {
  return xt::log(this->operands_[0]->forward());
}

template <typename T>
void LogOperator<T>::backward_pass(const Tensor<T>& grad) {
  std::cout << grad << std::endl;
  this->operands_[0]->backward(1.0 / this->operands_[0]->forward() * grad);
}

// Transpose operator.

template <typename T>
TrOperator<T>::TrOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
Tensor<T> TrOperator<T>::forward() {
  return xt::transpose(this->operands_[0]->forward());
}

template <typename T>
void TrOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(xt::transpose(grad));
}

// Sum operator.

template <typename T>
SumOperator<T>::SumOperator(const Operand<T> op)
    : UnaryOperator<T>({op}), axes_({}), keep_dim_(false) {}

template <typename T>
SumOperator<T>::SumOperator(const Operand<T> op, const std::vector<int>& axes,
                            const bool keep_dim)
    : UnaryOperator<T>({op}), axes_(axes), keep_dim_(keep_dim) {}

template <typename T>
Tensor<T> SumOperator<T>::forward() {
  auto value = this->operands_[0]->forward();
  Tensor<T> result;

  if (this->axes_.empty()) {
    result = xt::sum(value);
  } else {
    result = xt::sum(value, this->axes_);
  }

  if (!this->axes_.empty() && this->keep_dim_) {
    for (auto index : this->axes_) {
      result = xt::expand_dims(result, index);
    }
  }
  return result;
}

template <typename T>
void SumOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(
      xt::ones<T>(this->operands_[0]->forward().shape()) * grad);
}

// Mean operator.

template <typename T>
MeanOperator<T>::MeanOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
Tensor<T> MeanOperator<T>::forward() {
  return xt::mean(this->operands_[0]->forward());
}

template <typename T>
void MeanOperator<T>::backward_pass(const Tensor<T>& grad) {
  auto input_shape = this->operands_[0]->forward().shape();
  auto inv_n_elements =
      1.0 / std::accumulate(input_shape.begin(), input_shape.end(), 0);
  this->operands_[0]->backward(xt::ones<T>(input_shape) * inv_n_elements *
                               grad);
}

// Clamp operator.

template <typename T>
ClampOperator<T>::ClampOperator(const Operand<T> op, const T min_value,
                                const T max_value)
    : UnaryOperator<T>({op}), min_value_(min_value), max_value_(max_value) {}

template <typename T>
Tensor<T> ClampOperator<T>::forward() {
  return xt::clip(this->operands_[0]->forward(), this->min_value_,
                  this->max_value_);
}

template <typename T>
void ClampOperator<T>::backward_pass(const Tensor<T>& grad) {
  auto backward =
      xt::where((this->operands_[0]->forward() >= this->min_value_) &&
                    (this->operands_[0]->forward() <= this->max_value_),
                grad, xt::zeros_like(grad));

  std::cout << backward << std::endl;

  this->operands_[0]->backward(backward);
}

// Power operator.

template <typename T>
PowOperator<T>::PowOperator(const Operand<T> op, const T exponent)
    : UnaryOperator<T>({op}), exponent_(exponent) {}

template <typename T>
Tensor<T> PowOperator<T>::forward() {
  return xt::pow(this->operands_[0]->forward(), this->exponent_);
}

template <typename T>
void PowOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(
      this->exponent_ *
      xt::pow(this->operands_[0]->forward(), this->exponent_ - 1) * grad);
}  // TODO: implement cache for backward propagation.

// Max operator.

template <typename T>
MaxOperator<T>::MaxOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
MaxOperator<T>::MaxOperator(const Operand<T> op, const std::size_t axis,
                            const bool keep_dim)
    : UnaryOperator<T>({op}),
      axis_(axis),
      use_axis_(true),
      keep_dim_(keep_dim) {}

template <typename T>
Tensor<T> MaxOperator<T>::forward() {
  if (this->use_axis_) {
    if (this->keep_dim_) {
      return xt::expand_dims(xt::amax(this->operands_[0]->forward(),
                                      static_cast<int>(this->axis_)),
                             static_cast<int>(this->axis_));
    } else {
      return xt::amax(this->operands_[0]->forward(),
                      static_cast<int>(this->axis_));
    }
  } else {
    return xt::amax(this->operands_[0]->forward());
  }
}

template <typename T>
void MaxOperator<T>::backward_pass(const Tensor<T>& grad) {
  auto value = this->operands_[0]->forward();
  auto max_mask = xt::zeros_like(value);

  if (this->use_axis_) {
    auto max_idx = xt::argmax(value, static_cast<int>(this->axis_));
    // TODO: make this axis independent.
    max_mask[xt::all(), max_idx] = 1;
  } else {
    auto max_idx = xt::argmax(value);
    max_mask[max_idx] = 1.0;
  }

  this->operands_[0]->backward(max_mask * grad);
}

}  // namespace math
}  // namespace ado

#endif  // ADO_MATH_UNARY_OPERATORS_HPP