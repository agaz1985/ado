#ifndef ADO_CORE_MODEL_H
#define ADO_CORE_MODEL_H

#include "ado/types.h"

namespace ado {
namespace core {

class Model {
 public:
  virtual ~Model() = default;

  virtual void fit(const FloatTensor& x, const FloatTensor& y) = 0;
  virtual FloatTensor fit_predict(const FloatTensor& x, const FloatTensor& y) = 0;
  virtual FloatTensor predict(const FloatTensor& x) = 0;
  virtual FloatTensor decision_function(const FloatTensor& x) = 0;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_MODEL_H