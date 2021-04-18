#ifndef ADO_LAYERS_GRAPH_H
#define ADO_LAYERS_GRAPH_H

#include <map>
#include <string>

#include "ado/graph/operator.h"
#include "ado/layers/layer.h"

namespace ado {
namespace layers {

template <typename T>
class Graph {
 public:
  Operand<T> operator()(Operand<T> input) { return this->forward(input); }

  std::vector<typename Layer<T>::LayerParameters> parameters() {
    std::vector<typename Layer<T>::LayerParameters> params_list;
    for (auto layer : this->layers_) {
      if (!layer.second->parameters().empty()) {
        params_list.push_back(layer.second->parameters());
      }
    }
    return params_list; //TODO: add a way to save model.
  }

  void register_layer(typename Layer<T>::LayerPtr layer,
                      const std::string& name) {
    this->layers_[name] = layer;
  }

 protected:
  virtual Operand<T> forward(Operand<T> input) = 0;

  std::map<std::string, typename Layer<T>::LayerPtr> layers_;
};

}  // namespace layers
}  // namespace ado

#endif  // ADO_LAYERS_GRAPH_H
