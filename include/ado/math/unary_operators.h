#ifndef ADO_MATH_UNARY_OPERATORS_H
#define ADO_MATH_UNARY_OPERATORS_H

#include <limits>
#include <vector>

#include "ado/graph/operator.h"

namespace ado {
namespace math {

using ado::graph::Operand;
using ado::graph::UnaryOperator;

template <typename T>
class ExpOperator : public UnaryOperator<T> {
 public:
  ExpOperator(const Operand<T> op);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class LogOperator : public UnaryOperator<T> {
 public:
  LogOperator(const Operand<T> op);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class TrOperator : public UnaryOperator<T> {
 public:
  TrOperator(const Operand<T> op);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class SumOperator : public UnaryOperator<T> {
 public:
  SumOperator(const Operand<T> op);
  SumOperator(const Operand<T> op, const std::vector<int>& axes,
              const bool keep_dim = false);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;

 private:
  std::vector<int> axes_;
  bool keep_dim_ = false;
};

template <typename T>
class MeanOperator : public UnaryOperator<T> {
 public:
  MeanOperator(const Operand<T> op);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class ClampOperator : public UnaryOperator<T> {
 public:
  ClampOperator(const Operand<T> op,
                const T min_value = std::numeric_limits<T>::min(),
                const T max_value = std::numeric_limits<T>::max());

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;

 private:
  T min_value_ = std::numeric_limits<T>::min();
  T max_value_ = std::numeric_limits<T>::max();
};

template <typename T>
class PowOperator : public UnaryOperator<T> {
 public:
  PowOperator(const Operand<T> op, const T exponent);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;

 private:
  T exponent_ = 1.0;
};

template <typename T>
class MaxOperator : public UnaryOperator<T> {
 public:
  MaxOperator(const Operand<T> op);
  MaxOperator(const Operand<T> op, const std::size_t axis,
              const bool keep_dim = false);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;

 private:
  std::size_t axis_ = 0;
  bool use_axis_ = false;
  bool keep_dim_ = false;
};

}  // namespace math
}  // namespace ado

#include "ado/math/unary_operators.hpp"

#endif  // ADO_MATH_UNARY_OPERATORS_H