#ifndef ADO_CORE_MODEL_H
#define ADO_CORE_MODEL_H

#include "ado/types.h"

namespace ado {
namespace core {

class Model {
 public:
  virtual ~Model() = default;

  virtual void fit(const FloatArray& x, const FloatArray& y) = 0;
  virtual FloatArray fit_predict(const FloatArray& x, const FloatArray& y) = 0;
  virtual FloatArray predict(const FloatArray& x) = 0;
  virtual FloatArray decision_function(const FloatArray& x) = 0;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_MODEL_H