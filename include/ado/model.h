#ifndef ADO_MODEL_H
#define ADO_MODEL_H

#include "ado/types.h"

namespace ado {

class Model {
 public:
  virtual void fit(const FloatArray& x, const FloatArray& y) = 0;
  virtual FloatArray fit_predict(const FloatArray& x, const FloatArray& y) = 0;
  virtual FloatArray predict(const FloatArray& x) = 0;
  virtual FloatArray prob(const FloatArray& x) = 0;
};

}  // namespace ado

#endif  // ADO_MODEL_H