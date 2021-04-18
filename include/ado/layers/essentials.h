#ifndef ADO_LAYERS_ESSENTIALS_H
#define ADO_LAYERS_ESSENTIALS_H

#include "ado/graph/variable.h"
#include "ado/layers/layer.h"
#include "ado/math/functions.h"
#include "ado/types.h"

namespace ado {
namespace layers {
namespace essentials {

using ado::graph::Variable;
using ado::math::dot;
using ado::math::tr;
using ado::math::operator+;
using ado::layers::Layer;

template <typename T>
class Linear : public Layer<T> {
 public:
  Linear(const std::size_t num_input_features,
         const std::size_t num_output_features, const bool use_bias = true)
      : Layer<T>(LayerType::Linear, num_input_features, num_output_features),
        use_bias_(use_bias) {
    this->parameters_["weights"] = std::make_shared<Variable<T>>(
        Tensor<T>({this->num_output_features(), this->num_input_features()}, 0),
        true);
    this->parameters_["bias"] = std::make_shared<Variable<T>>(
        Tensor<T>({1, this->num_output_features()}, 0), true);

    // Initialize the layer's parameters.
    this->init_params();
  }

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto y = dot(input, tr(this->parameters_["weights"]));

    if (this->use_bias_) {
      return y + this->parameters_["bias"];
    } else {
      return y;
    }
  }

 private:
  void init_params() {  // TODO: move generator type out of variable and make it
                        // as input to this layer.
    this->parameters_["weights"]
        ->rand();  // TODO: implememnt
                   // setRandom(core::graph::RandomGeneratorType::Normal);
    this->parameters_["bias"]->zeros();
  }

  bool use_bias_ = true;
};

}  // namespace essentials
}  // namespace layers
}  // namespace ado

#endif  // ADO_LAYERS_ESSENTIALS_H
