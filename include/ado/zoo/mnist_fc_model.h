#ifndef ADO_ZOO_MNIST_FC_MODEL_H
#define ADO_ZOO_MNIST_FC_MODEL_H

#include "ado/graph/variable.h"
#include "ado/layers/activations.h"
#include "ado/layers/essentials.h"
#include "ado/layers/graph.h"

using ado::layers::Graph;

using ado::graph::Operand;
using ado::layers::activations::ReLU;
using ado::layers::activations::Sigmoid;
using ado::layers::essentials::Linear;

// TODO: move h to hpp implementations.
// TOOD: make layer creation easier (no make_shared)

namespace ado {
namespace zoo {

template <typename T>
class MNISTFCModel : public Graph<T> {
 public:
  MNISTFCModel(const std::size_t input_size, const std::size_t hidden_size,
               const std::size_t num_classes) {
    auto fc_1_ = std::make_shared<Linear<T>>(input_size, hidden_size, true);
    auto relu_ = std::make_shared<ReLU<T>>();
    auto fc_2_ = std::make_shared<Linear<T>>(hidden_size, num_classes, true);
    auto act_ = std::make_shared<Sigmoid<T>>();

    this->register_layer(fc_1_, "fc_1_");
    this->register_layer(relu_, "relu_");
    this->register_layer(fc_2_, "fc_2_");
    this->register_layer(act_, "act_");
  }

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto x = (*this->layers_["fc_1_"])(input);
    x = (*this->layers_["relu_"])(x);
    x = (*this->layers_["fc_2_"])(x);
    return (*this->layers_["act_"])(x);
  }
};

}  // namespace zoo
}  // namespace ado

#endif  // ADO_ZOO_MNIST_FC_MODEL_H