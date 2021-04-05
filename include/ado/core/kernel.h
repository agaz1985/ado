#ifndef ADO_CORE_KERNEL_H
#define ADO_CORE_KERNEL_H

#include "ado/types.h"

namespace ado {
namespace core {

enum class KernelType { Polynomial = 0, RBF = 1, Sigmoid = 2 };

class Kernel {
 public:
  virtual ~Kernel() = default;

  Kernel(const KernelType type) : _type(type){};
  virtual Float operator()(const FloatArray& x1,
                                const FloatArray& x2) const = 0;
  inline KernelType type() const { return this->_type; }

 private:
  KernelType _type;
};

class KernelPolynomial : public Kernel {
 public:
  explicit KernelPolynomial(const Float degree, const Float alpha,
                            const Float bias);
  virtual Float operator()(const FloatArray& x1, const FloatArray& x2) const override;

 private:
  Float _degree = 1.0;
  Float _alpha = 1.0;
  Float _bias = 0.0;
};

class KernelRBF : public Kernel {
 public:
  explicit KernelRBF(const Float sigma);
  virtual Float operator()(const FloatArray& x1, const FloatArray& x2) const override;

 private:
  Float _sigma = 1.0;
};

class KernelSigmoid : public Kernel {
 public:
  explicit KernelSigmoid(const Float alpha, const Float bias);
  virtual Float operator()(const FloatArray& x1, const FloatArray& x2) const override;

 private:
  Float _alpha = 1.0;
  Float _bias = 0.0;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_KERNEL_H