#ifndef ADO_CORE_LINEAR_SVM_MODEL_H
#define ADO_CORE_LINEAR_SVM_MODEL_H

#include <memory>

#include "ado/graph/variable.h"
#include "ado/layers/activations.h"
#include "ado/layers/essentials.h"
#include "ado/layers/graph.h"

using ado::graph::Node;
using ado::layers::Graph;

using ado::graph::Operand;
using ado::layers::essentials::Linear;

namespace ado {
namespace core {

template <typename T>
class LinearSVMModel : public Graph<T> {
 public:
  LinearSVMModel(const std::size_t input_size, const bool bias = false) {
    auto fc_ = std::make_shared<Linear<T>>(input_size, 1, bias);
    this->register_layer(fc_, "fc_");
  }

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    return (*this->layers_["fc_"])(input);
  }
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_LINEAR_SVM_MODEL_H