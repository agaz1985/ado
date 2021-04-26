#ifndef ADO_OPTIMIZERS_OPTIMIZER_H
#define ADO_OPTIMIZERS_OPTIMIZER_H

#include <memory>
#include <string>
#include <vector>

#include "ado/layers/layer.h"

namespace ado {
namespace optimizers {

using ado::layers::Layer;

template <typename T>
class Optimizer {
 public:
  Optimizer(const std::vector<typename Layer<T>::LayerParameters>& parameters)
      : parameters_(parameters) {}

  void step() {
    for (auto layer_params : this->parameters_) {
      for (auto parameter : layer_params) {
        if (parameter.second->requires_grad()) {
          parameter.second->update(compute_update(
              parameter.second->forward(), parameter.second->last_update(),
              parameter.second->grad()));
        }
      }
    }
  }

  void zero_grad() {
    for (auto layer_params : this->parameters_) {
      for (auto parameter : layer_params) {
        if (parameter.second->requires_grad()) {
          parameter.second->zero_grad();
        }
      }
    }
  }

 protected:
  virtual Tensor<T> compute_update(const Tensor<T>& params,
                                   const Tensor<T>& prev_update,
                                   const Tensor<T>& grad) = 0;

 private:
  std::vector<typename Layer<T>::LayerParameters> parameters_;
};

template <typename T>
class SGD : public Optimizer<T> {
 public:
  SGD(const std::vector<typename Layer<T>::LayerParameters>& parameters,
      const float lr = 1e-3, const float momentum = 0.9,
      const float decay = 0.0)
      : Optimizer<T>(parameters), lr_(lr), momentum_(momentum) {}

 protected:
  virtual Tensor<T> compute_update(const Tensor<T>& params,
                                   const Tensor<T>& prev_update,
                                   const Tensor<T>& grad) override {
    return this->momentum_ * prev_update +
           this->lr_ * (grad + this->decay_ * params);
  }

 private:
  float lr_ = 1e-3;
  float momentum_ = 0.0;
  float decay_ = 0.0;
};

}  // namespace optimizers
}  // namespace ado

#endif  // ADO_OPTIMIZERS_OPTIMIZER_H
