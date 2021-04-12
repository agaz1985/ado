#ifndef ADO_CORE_LINEAR_SVM_H
#define ADO_CORE_LINEAR_SVM_H

#include <memory>

#include "ado/core/model.h"
#include "ado/layers/graph.h"
#include "ado/types.h"

namespace ado {
namespace core {

using ado::layers::Graph;

class LinearSVM : public Model {
 public:
  LinearSVM(const Float C, const Float lr, const Float momentum,
            const Float decay, const Int epochs, const bool intercept,
            const std::size_t seed);

  LinearSVM() = default;

  void fit(const FloatTensor& x, const FloatTensor& y) override;

  FloatTensor fit_predict(const FloatTensor& x, const FloatTensor& y) override;

  FloatTensor predict(const FloatTensor& x) override;

  FloatTensor decision_function(const FloatTensor& x) override;

 private:
  Float C_ = 1.0;
  Float lr_ = 1e-2;
  Float momentum_ = 0.0;
  Float decay_ = 0.0;
  Int epochs_ = 1e3;
  bool intercept_ = true;
  std::size_t seed_ = 16;
  std::unique_ptr<Graph<Float>> _model = nullptr;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_LINEAR_SVM_H
