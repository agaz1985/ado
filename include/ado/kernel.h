#ifndef ADO_KERNEL_H
#define ADO_KERNEL_H

#include "ado/types.h"

namespace ado {

enum class KernelType { Linear = 0, RBF = 1 };

class Kernel {
 public:
  Kernel(const KernelType type) : _type(type){};
  virtual Float operator()(const FloatArray& x1,
                           const FloatArray& x2) const = 0;
  inline KernelType type() const { return this->_type; }

 private:
  KernelType _type;
};

class KernelLinear : public Kernel {
 public:
  KernelLinear();
  Float operator()(const FloatArray& x1, const FloatArray& x2) const override;
};

class KernelRBF : public Kernel {
 public:
  explicit KernelRBF(const Float sigma);
  Float operator()(const FloatArray& x1, const FloatArray& x2) const override;

 private:
  Float _sigma = 5.0;
};

}  // namespace ado

#endif  // ADO_KERNEL_H