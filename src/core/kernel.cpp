#include "ado/core/kernel.h"

#include <xtensor-blas/xlinalg.hpp>

namespace ado {
namespace core {

// Polynomial Kernel.

KernelPolynomial::KernelPolynomial(const Float degree, const Float alpha,
                                   const Float bias)
    : Kernel(KernelType::Polynomial),
      _degree(degree),
      _alpha(alpha),
      _bias(bias){};

Float KernelPolynomial::operator()(const FloatArray& x1,
                                   const FloatArray& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::pow(this->_alpha * s + this->_bias, this->_degree)(0);
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

// Sigmoid Kernel.

KernelSigmoid::KernelSigmoid(const Float alpha, const Float bias)
    : Kernel(KernelType::Sigmoid), _alpha(alpha), _bias(bias){};

Float KernelSigmoid::operator()(const FloatArray& x1,
                                const FloatArray& x2) const {
  auto s = xt::linalg::dot(x1, xt::transpose(x2));
  return xt::tanh(this->_alpha * s + this->_bias)(0);
}

}  // namespace core
}  // namespace ado