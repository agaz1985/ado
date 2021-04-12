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

FloatTensor KernelPolynomial::operator()(const FloatTensor& x1,
                                        const FloatTensor& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::pow(this->_gamma * s + this->_coeff, this->_degree);
}

// RBF Kernel.

KernelRBF::KernelRBF(const Float gamma)
    : Kernel(KernelType::RBF), _gamma(gamma) {}

FloatTensor KernelRBF::operator()(const FloatTensor& x1,
                                 const FloatTensor& x2) const {
  auto distance = xt::sum(xt::pow(xt::abs(x1 - x2), 2), -1);
  return xt::exp(-this->_gamma * distance);
}

// Sigmoid Kernel.

KernelSigmoid::KernelSigmoid(const Float gamma, const Float coeff)
    : Kernel(KernelType::Sigmoid), _gamma(gamma), _coeff(coeff) {}

FloatTensor KernelSigmoid::operator()(const FloatTensor& x1,
                                     const FloatTensor& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::tanh(this->_gamma * s + this->_coeff);
}

}  // namespace core
}  // namespace ado