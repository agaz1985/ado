#ifndef ADO_LAYERS_ACTIVATIONS_H
#define ADO_LAYERS_ACTIVATIONS_H

#include <limits>

#include "ado/graph/operator.h"
#include "ado/layers/layer.h"
#include "ado/math/functions.h"

namespace ado {
namespace layers {
namespace activations {

using ado::math::operator-;
using ado::math::operator+;
using ado::math::operator*;
using ado::math::operator/;
using ado::math::clamp;
using ado::math::exp;
using ado::math::sum;

using ado::graph::Operand;

template <typename T>
class Sigmoid : public Layer<T> {
 public:
  Sigmoid() : Layer<T>(LayerType::Sigmoid) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    return 1 / (1 + exp(-1 * input));
  }
};

template <typename T>
class Tanh : public Layer<T> {
 public:
  Tanh() : Layer<T>(LayerType::Tanh) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto e_x = exp(input);
    auto e_n_x = exp(-1 * input);
    return (e_x - e_n_x) / (e_x + e_n_x);
  }
};

template <typename T>
class ReLU : public Layer<T> {
 public:
  ReLU() : Layer<T>(LayerType::ReLU) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    return clamp(input, 0.0);
  }
};

}  // namespace activations
}  // namespace layers
}  // namespace ado

#endif  // ADO_LAYERS_ACTIVATIONS_H
