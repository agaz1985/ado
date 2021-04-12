#ifndef ADO_LAYERS_LAYER_H
#define ADO_LAYERS_LAYER_H

#include <map>
#include <memory>

#include "ado/graph/operator.h"
#include "ado/graph/variable.h"

namespace ado {
namespace layers {

using ado::graph::Operand;
using ado::graph::Variable;

enum class LayerType {
  Linear = 0,
  Sigmoid = 1,
  Tanh = 2,
  ReLU = 3,
  Softmax = 4
};

template <typename T>
class Layer {
 public:
  using LayerPtr = std::shared_ptr<Layer<T>>;  // TODO: make this consistent.
  using LayerParameters =
      std::map<std::string, typename Variable<T>::VariablePtr>;

  Layer(const LayerType type, const std::size_t num_input_features = 0,
        const std::size_t num_output_features = 0)
      : type_(type),
        num_input_features_(num_input_features),
        num_output_features_(num_output_features) {}

  Operand<T> operator()(Operand<T> input) { return this->forward(input); }

  LayerParameters parameters() { return this->parameters_; }

  LayerType type() const { return this->type_; }

  std::size_t num_input_features() const { return this->num_input_features_; }

  std::size_t num_output_features() const { return this->num_output_features_; }

 protected:
  virtual Operand<T> forward(Operand<T> input) = 0;

  LayerType type_;
  LayerParameters parameters_;
  std::size_t num_input_features_ = 0;
  std::size_t num_output_features_ = 0;
};

}  // namespace layers
}  // namespace ado

#endif  // ADO_LAYERS_LAYER_H
