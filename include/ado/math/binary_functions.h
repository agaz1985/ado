#ifndef ADO_MATH_BINARY_FUNCTIONS_H
#define ADO_MATH_BINARY_FUNCTIONS_H

#include "ado/graph/operator.h"
#include "ado/graph/variable.h"
#include "ado/math/binary_operators.h"

namespace ado {
namespace math {

using ado::graph::Operand;
using ado::graph::Variable;

// Functions.

// Add functions.

template <typename T>
Operand<T> add(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<AddOperator<T>>(op1, op2);
}

template <typename T>
Operand<T> add(const T op1, const Operand<T> op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op1), false);
  return std::make_shared<AddOperator<T>>(tmp, op2);
}

template <typename T>
Operand<T> add(const Operand<T> op1, const T op2) {
  return add<T>(op2, op1);
}

// Sub functions.

template <typename T>
Operand<T> sub(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<SubOperator<T>>(op1, op2);
}

template <typename T>
Operand<T> sub(const T op1, const Operand<T> op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op1), false);
  return std::make_shared<SubOperator<T>>(tmp, op2);
}

template <typename T>
Operand<T> sub(const Operand<T> op1, const T op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op2), false);
  return std::make_shared<SubOperator<T>>(op1, tmp);
}

// Mul functions.

template <typename T>
Operand<T> mul(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<MulOperator<T>>(op1, op2);
}

template <typename T>
Operand<T> mul(const T op1, const Operand<T> op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op1), false);
  return std::make_shared<MulOperator<T>>(tmp, op2);
}

template <typename T>
Operand<T> mul(const Operand<T> op1, const T op2) {
  return mul<T>(op2, op1);
}

// Div functions.

template <typename T>
Operand<T> div(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<DivOperator<T>>(op1, op2);
}

template <typename T>
Operand<T> div(const T op1, const Operand<T> op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op1), false);
  return std::make_shared<DivOperator<T>>(tmp, op2);
}

template <typename T>
Operand<T> div(const Operand<T> op1, const T op2) {
  auto tmp = std::make_shared<Variable<T>>(Tensor<T>(op2), false);
  return std::make_shared<DivOperator<T>>(op1, tmp);
}

// Dot functions.

template <typename T>
Operand<T> dot(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<DotProdOperator<T>>(op1, op2);
}

// Maximum functions.

template <typename T>
Operand<T> maximum(const Operand<T> op1, const Operand<T> op2) {
  return std::make_shared<MaximumOperator<T>>(op1, op2);
}

template <typename T>
Operand<T> maximum(const T op1, const Operand<T> op2) {
  auto tmp = std::make_shared<Variable<T>>(xt::zeros_like(op2->forward()) + op1,
                                           false);
  return std::make_shared<MaximumOperator<T>>(tmp, op2);
}

template <typename T>
Operand<T> maximum(const Operand<T> op1, const T op2) {
  auto tmp = std::make_shared<Variable<T>>(xt::zeros_like(op1->forward()) + op2,
                                           false);
  return std::make_shared<MaximumOperator<T>>(op1, tmp);
}

// Float function operators.

// Add float functions.

Operand<Float> operator+(const Operand<Float> op1, const Operand<Float> op2) {
  return add<Float>(op1, op2);
}

Operand<Float> operator+(const Float op1, const Operand<Float> op2) {
  return add<Float>(op1, op2);
}

Operand<Float> operator+(const Operand<Float> op1, const Float op2) {
  return add<Float>(op1, op2);
}

// Sub float functions.

Operand<Float> operator-(const Operand<Float> op1, const Operand<Float> op2) {
  return sub<Float>(op1, op2);
}

Operand<Float> operator-(const Float op1, const Operand<Float> op2) {
  return sub<Float>(op1, op2);
}

Operand<Float> operator-(const Operand<Float> op1, const Float op2) {
  return sub<Float>(op1, op2);
}

// Mul float functions.

Operand<Float> operator*(const Operand<Float> op1, const Operand<Float> op2) {
  return mul<Float>(op1, op2);
}

Operand<Float> operator*(const Float op1, const Operand<Float> op2) {
  return mul<Float>(op1, op2);
}

Operand<Float> operator*(const Operand<Float> op1, const Float op2) {
  return mul<Float>(op1, op2);
}

// Div float functions.

Operand<Float> operator/(const Operand<Float> op1, const Operand<Float> op2) {
  return div<Float>(op1, op2);
}

Operand<Float> operator/(const Float op1, const Operand<Float> op2) {
  return div<Float>(op1, op2);
}

Operand<Float> operator/(const Operand<Float> op1, const Float op2) {
  return div<Float>(op1, op2);
}

// Dot product float functions.

Operand<Float> dot(const Operand<Float> op1, const Operand<Float> op2) {
  return dot<Float>(op1, op2);
}

// Maximum float functions.

Operand<Float> maximum(const Operand<Float> op1, const Operand<Float> op2) {
  return maximum<Float>(op1, op2);
}

Operand<Float> maximum(const Float op1, const Operand<Float> op2) {
  return maximum<Float>(op1, op2);
}

Operand<Float> maximum(const Operand<Float> op1, const Float op2) {
  return maximum<Float>(op1, op2);
}

}  // namespace math
}  // namespace ado

#endif  // ADO_MATH_BINARY_FUNCTIONS_H