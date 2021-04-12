#ifndef ADO_MATH_UNARY_FUNCTIONS_H
#define ADO_MATH_UNARY_FUNCTIONS_H

#include <limits>

#include "ado/graph/operator.h"
#include "ado/math/unary_operators.h"

namespace ado {
namespace math {

using ado::graph::Operand;

// Functions.

template <typename T>
Operand<T> exp(const Operand<T> op) {
  return std::make_shared<ExpOperator<T>>(op);
}

template <typename T>
Operand<T> log(const Operand<T> op) {
  return std::make_shared<LogOperator<T>>(op);
}

template <typename T>
Operand<T> tr(const Operand<T> op) {
  return std::make_shared<TrOperator<T>>(op);
}

template <typename T>
Operand<T> sum(const Operand<T> op) {
  return std::make_shared<SumOperator<T>>(op);
}

template <typename T>
Operand<T> mean(const Operand<T> op) {
  return std::make_shared<MeanOperator<T>>(op);
}

template <typename T>
Operand<T> clamp(const Operand<T> op,
                 const T min_value = std::numeric_limits<T>::min(),
                 const T max_value = std::numeric_limits<T>::max()) {
  return std::make_shared<ClampOperator<T>>(op, min_value, max_value);
}

template <typename T>
Operand<T> pow(const Operand<T> op, const T exponent) {
  return std::make_shared<PowOperator<T>>(op, exponent);
}

// Float function operators.

Operand<Float> exp(const Operand<Float> op) { return exp<Float>(op); }

Operand<Float> log(const Operand<Float> op) { return log<Float>(op); }

Operand<Float> tr(const Operand<Float> op) { return tr<Float>(op); }

Operand<Float> sum(const Operand<Float> op) { return sum<Float>(op); }

Operand<Float> clamp(
    const Operand<Float> op,
    const Float min_value = std::numeric_limits<Float>::min(),
    const Float max_value = std::numeric_limits<Float>::max()) {
  return clamp<Float>(op, min_value, max_value);
}

Operand<Float> pow(const Operand<Float> op, const Float exponent) {
  return pow<Float>(op, exponent);
}

}  // namespace math
}  // namespace ado

#endif  // ADO_MATH_UNARY_FUNCTIONS_H