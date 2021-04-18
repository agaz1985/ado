#ifndef ADO_MATH_BINARY_OPERATORS_H
#define ADO_MATH_BINARY_OPERATORS_H

#include "ado/graph/operator.h"

namespace ado {
namespace math {

using ado::graph::BinaryOperator;
using ado::graph::Operand;

template <typename T>
class AddOperator : public BinaryOperator<T> {
 public:
  AddOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class SubOperator : public BinaryOperator<T> {
 public:
  SubOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class MulOperator : public BinaryOperator<T> {
 public:
  MulOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class DivOperator : public BinaryOperator<T> {
 public:
  DivOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class DotProdOperator : public BinaryOperator<T> {
 public:
  DotProdOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class MaximumOperator : public BinaryOperator<T> {
 public:
  MaximumOperator(const Operand<T> op1, const Operand<T> op2);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

template <typename T>
class WhereOperator : public BinaryOperator<T> {
 public:
  WhereOperator(const Operand<T> op1, const Operand<T> op2,
                const Tensor<bool> condition);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;

 private:
  Tensor<bool> condition_ = T();
};

}  // namespace math
}  // namespace ado

#include "ado/math/binary_operators.hpp"

#endif  // ADO_MATH_BINARY_OPERATORS_H