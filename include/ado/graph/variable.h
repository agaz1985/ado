#ifndef ADO_GRAPH_VARIABLE_H
#define ADO_GRAPH_VARIABLE_H

#include "ado/graph/node.h"
#include "ado/random/seed.h"
#include "ado/types.h"

namespace ado {
namespace graph {

template <typename T>
class Variable : public Tensor<T>, public Node<Tensor<T>> {
 public:
  using VariablePtr = std::shared_ptr<Variable<T>>;

  explicit Variable(const Variable<T>& variable);
  Variable(const Tensor<T>& tensor, const bool requires_gradient = false);
  Variable(const TensorShape<T>& shape, const T value = 0,
           const bool requires_gradient = false);

  void zeros();
  void ones();
  void update(const Tensor<T>& value);

  virtual Tensor<T> forward() override;

 protected:
  virtual void backward_pass(const Tensor<T>& grad) override;
};

using FloatVariable = Variable<Float>;
using IntVariable = Variable<Int>;

}  // namespace graph
}  // namespace ado

#include "ado/graph/variable.hpp"

#endif  // ADO_GRAPH_VARIABLE_H
