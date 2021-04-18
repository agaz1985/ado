#ifndef ADO_MATH_BINARY_OPERATORS_HPP
#define ADO_MATH_BINARY_OPERATORS_HPP

#include <xtensor-blas/xlinalg.hpp>

#include "ado/math/binary_operators.h"

namespace ado {
namespace math {

// Add operator.

template <typename T>
AddOperator<T>::AddOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> AddOperator<T>::forward() {
  return this->operands_[0]->forward() + this->operands_[1]->forward();
}

template <typename T>
void AddOperator<T>::backward_pass(const Tensor<T>& grad) {
  if (this->operands_[0]->forward().shape() != grad.shape()) {
    this->operands_[0]->backward(
        xt::row(grad, 0));  // TODO: check this, create a function. This is
                            // caused by broadcast.
  } else {
    this->operands_[0]->backward(grad);
  }

  if (this->operands_[1]->forward().shape() != grad.shape()) {
    this->operands_[1]->backward(xt::row(grad, 0));
  } else {
    this->operands_[1]->backward(grad);
  }
}

// Sub operator.

template <typename T>
SubOperator<T>::SubOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> SubOperator<T>::forward() {
  return this->operands_[0]->forward() - this->operands_[1]->forward();
}

template <typename T>
void SubOperator<T>::backward_pass(const Tensor<T>& grad) {
  if (this->operands_[0]->forward().shape() != grad.shape()) {
    this->operands_[0]->backward(xt::row(grad, 0));
  } else {
    this->operands_[0]->backward(grad);
  }

  if (this->operands_[1]->forward().shape() != grad.shape()) {
    this->operands_[1]->backward(xt::row(-grad, 0));
  } else {
    this->operands_[1]->backward(-grad);
  }
}

// Mul operator.

template <typename T>
MulOperator<T>::MulOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> MulOperator<T>::forward() {
  return this->operands_[0]->forward() * this->operands_[1]->forward();
}

template <typename T>
void MulOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(this->operands_[1]->forward() * grad);
  this->operands_[1]->backward(this->operands_[0]->forward() * grad);
}

// Div operator.

template <typename T>
DivOperator<T>::DivOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> DivOperator<T>::forward() {
  return this->operands_[0]->forward() / this->operands_[1]->forward();
}

template <typename T>
void DivOperator<T>::backward_pass(const Tensor<T>& grad) {
  const auto squared_denominator = xt::pow(this->operands_[1]->forward(), 2);
  this->operands_[0]->backward(
      (this->operands_[1]->forward() / squared_denominator) * grad);
  this->operands_[1]->backward(
      (-this->operands_[0]->forward() / squared_denominator) * grad);
}

// Dot prod. operator.

template <typename T>
DotProdOperator<T>::DotProdOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> DotProdOperator<T>::forward() {
  return xt::linalg::dot(this->operands_[0]->forward(),
                         this->operands_[1]->forward());
}

template <typename T>
void DotProdOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(
      xt::linalg::dot(grad, xt::transpose(this->operands_[1]->forward())));
  this->operands_[1]->backward(
      xt::linalg::dot(xt::transpose(this->operands_[0]->forward()), grad));
}

// Maximum operator.

template <typename T>
MaximumOperator<T>::MaximumOperator(const Operand<T> op1, const Operand<T> op2)
    : BinaryOperator<T>({op1, op2}) {}

template <typename T>
Tensor<T> MaximumOperator<T>::forward() {
  return xt::maximum(this->operands_[0]->forward(),
                     this->operands_[1]->forward());
}

template <typename T>
void MaximumOperator<T>::backward_pass(const Tensor<T>& grad) {
  auto zeros = xt::zeros<double>(grad.shape());
  this->operands_[0]->backward(
      xt::where(this->operands_[0]->forward() >= this->operands_[1]->forward(),
                grad, zeros));
  this->operands_[1]->backward(
      xt::where(this->operands_[1]->forward() > this->operands_[0]->forward(),
                grad, zeros));
}

// Where operator.

template <typename T>
WhereOperator<T>::WhereOperator(const Operand<T> op1, const Operand<T> op2,
                                const Tensor<bool> condition)
    : BinaryOperator<T>({op1, op2}), condition_(condition) {}

template <typename T>
Tensor<T> WhereOperator<T>::forward() {
  return xt::where(this->condition_, this->operands_[0]->forward(),
                   this->operands_[1]->forward());
}

template <typename T>
void WhereOperator<T>::backward_pass(const Tensor<T>& grad) {
  this->operands_[0]->backward(
      xt::where(this->condition_, grad, xt::zeros_like(grad)));
  this->operands_[1]->backward(
      xt::where(this->condition_, xt::zeros_like(grad), grad));
}

}  // namespace math
}  // namespace ado

#endif  // ADO_MATH_BINARY_OPERATORS_HPP