#ifndef ADO_MATH_UNARY_OPERATORS_HPP
#define ADO_MATH_UNARY_OPERATORS_HPP

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
  this->operands_[0]->backward((1 / this->operands_[0]->forward()) * grad);
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
  this->operands_[0]->backward(
      xt::ones<T>(xt::transpose(this->operands_[0]->forward()).shape()) * grad);
}

// Sum operator.

template <typename T>
SumOperator<T>::SumOperator(const Operand<T> op) : UnaryOperator<T>({op}) {}

template <typename T>
Tensor<T> SumOperator<T>::forward() {
  return xt::sum(this->operands_[0]->forward());
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
  auto inv_n_elements = 1.0 / std::accumulate(input_shape.begin(), input_shape.end(), 0);
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
  auto higher_indexes =
      xt::argwhere(this->operands_[0]->forward() >= this->max_value_);
  auto lower_indexes =
      xt::argwhere(this->operands_[0]->forward() <= this->min_value_);

  auto idx = xt::from_indices(higher_indexes);
  auto tmp = xt::ones<T>(this->operands_[0]->forward().shape());

  // TODO: set high and low tmp elements to 0.
  this->operands_[0]->backward(tmp * grad);
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

}  // namespace math
}  // namespace ado

#endif  // ADO_MATH_UNARY_OPERATORS_HPP