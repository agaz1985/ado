#include "ado/kernel.h"

#include <xtensor-blas/xlinalg.hpp>

namespace ado {

// Linear Kernel.

KernelLinear::KernelLinear() : Kernel(KernelType::Linear){};

Float KernelLinear::operator()(const FloatArray& x1,
                               const FloatArray& x2) const {
  return xt::linalg::dot(x1, xt::transpose(x2))(0);
}

// RBF Kernel.

KernelRBF::KernelRBF(const Float sigma)
    : Kernel(KernelType::RBF), _sigma(sigma){};

Float KernelRBF::operator()(const FloatArray& x1, const FloatArray& x2) const {
  FloatArray x2_t = xt::transpose(x2);
  auto s = xt::linalg::dot(x1, xt::transpose(x1)) + xt::linalg::dot(x2, x2_t) -
           2 * xt::linalg::dot(x1, x2_t);
  return xt::exp(-s / (2 * std::pow(this->_sigma, 2)))(0);
}

}  // namespace ado