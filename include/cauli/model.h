#include "cauli/types.h"

class Model {
 public:
  virtual void fit(const FloatArray& x, const FloatArray& y) = 0;
  virtual FloatArray predict(const FloatArray& x) = 0;
};