#ifndef ADO_LAYERS_ACTIVATIONS_H
#define ADO_LAYERS_ACTIVATIONS_H

#include <limits>

#include "ado/constants.h"
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
using ado::math::cond;
using ado::math::exp;
using ado::math::max;
using ado::math::sum;

using ado::MAX_FLOAT_VALUE;
using ado::MIN_FLOAT_VALUE;

using ado::graph::Operand;

template <typename T>
class Sigmoid : public Layer<T> {
 public:
  Sigmoid() : Layer<T>(LayerType::Sigmoid) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto exponential = exp(-1.0 * input);
    auto c_exponential = clamp(exponential, MIN_FLOAT_VALUE, MAX_FLOAT_VALUE);
    return 1.0 / (1.0 + c_exponential);
  }
};

template <typename T>
class Softmax : public Layer<T> {
 public:
  Softmax() : Layer<T>(LayerType::Softmax) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto shifted_input = input - max(input, 1, true);
    auto exponential = exp(shifted_input);
    return exponential / sum(exponential, 1, true);
  }
};

template <typename T>
class Tanh : public Layer<T> {
 public:
  Tanh() : Layer<T>(LayerType::Tanh) {}

 protected:
  virtual Operand<T> forward(Operand<T> input) override {
    auto exponential = exp(-2.0 * input);
    auto c_exponential = clamp(exponential, MIN_FLOAT_VALUE, MAX_FLOAT_VALUE);
    auto denominator = 1.0 + c_exponential;
    return (2.0 / denominator) - 1;
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
