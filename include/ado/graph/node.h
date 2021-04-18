#ifndef ADO_GRAPH_NODE_H
#define ADO_GRAPH_NODE_H

#include "ado/types.h"

namespace ado {
namespace graph {

template <typename T>
class Node {
 public:
  explicit Node(const bool requires_grad = false, const T grad = T());
  virtual ~Node() = default;

  virtual T forward() = 0;

  void backward();
  void backward(const T& grad);

  bool requires_grad() const;
  void update_grad(const T& grad);
  void zero_grad();

  T grad() const;

 protected:
  virtual void backward_pass(const T& grad) = 0;

  T grad_;
  bool requires_grad_ = false;
  bool empty_grad_ = true;
};

template <typename T>
using TensorNode = Node<Tensor<T>>;

}  // namespace graph
}  // namespace ado

#include "ado/graph/node.hpp"

#endif  // ADO_GRAPH_NODE_H