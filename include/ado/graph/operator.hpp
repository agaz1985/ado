#ifndef ADO_GRAPH_OPERATOR_HPP
#define ADO_GRAPH_OPERATOR_HPP

#include "ado/graph/node.h"
#include "ado/graph/operator.h"

namespace ado {
namespace graph {

template <typename T, std::size_t NumOperands>
Operator<T, NumOperands>::Operator(const OperandList& operands)
    : Node<Tensor<T>>(true), operands_(operands) {}

}  // namespace graph
}  // namespace ado

#endif  // ADO_GRAPH_OPERATOR_HPP