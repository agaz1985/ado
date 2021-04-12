#ifndef ADO_GRAPH_OPERATOR_H
#define ADO_GRAPH_OPERATOR_H

#include <array>

#include "ado/graph/node.h"
#include "ado/types.h"

namespace ado {
namespace graph {

template <typename T>
using Operand = std::shared_ptr<TensorNode<T>>;

template <typename T, std::size_t NumOperands>
class Operator : public Node<Tensor<T>> {
 public:
  using OperandList = std::array<Operand<T>, NumOperands>;

  explicit Operator(const OperandList& operands);

 protected:
  OperandList operands_;
};

template <typename T>
using UnaryOperator = Operator<T, 1>;

template <typename T>
using BinaryOperator = Operator<T, 2>;

}  // namespace graph
}  // namespace ado

#include "ado/graph/operator.hpp"

#endif  // ADO_GRAPH_OPERATOR_H