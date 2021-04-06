#include "ado/core/kernel.h"

#include <xtensor-blas/xlinalg.hpp>

namespace ado {
namespace core {

// Polynomial Kernel.

KernelPolynomial::KernelPolynomial(const Float degree, const Float gamma,
                                   const Float coeff)
    : Kernel(KernelType::Polynomial),
      _degree(degree),
      _gamma(gamma),
      _coeff(coeff) {}

FloatArray KernelPolynomial::operator()(const FloatArray& x1,
                                        const FloatArray& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::pow(this->_gamma * s + this->_coeff, this->_degree);
}

// RBF Kernel.

KernelRBF::KernelRBF(const Float gamma)
    : Kernel(KernelType::RBF), _gamma(gamma) {}

FloatArray KernelRBF::operator()(const FloatArray& x1,
                                 const FloatArray& x2) const {
  FloatArray x2_t = xt::transpose(x2);
  auto distance = xt::linalg::dot(x1, xt::transpose(x1)) +
                  xt::linalg::dot(x2, x2_t) - 2 * xt::linalg::dot(x1, x2_t);
  return xt::exp(-distance / (2 * std::pow(this->_gamma, 2)));
}

// Sigmoid Kernel.

KernelSigmoid::KernelSigmoid(const Float gamma, const Float coeff)
    : Kernel(KernelType::Sigmoid), _gamma(gamma), _coeff(coeff) {}

FloatArray KernelSigmoid::operator()(const FloatArray& x1,
                                     const FloatArray& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::tanh(this->_gamma * s + this->_coeff);
}

}  // namespace core
}  // namespace ado