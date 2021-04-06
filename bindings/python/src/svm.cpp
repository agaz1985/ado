#include "svm.h"

namespace {
using ado::core::KernelType;

const std::map<std::string, KernelType> KERNEL_MAP = {
    {"rbf", KernelType::RBF},
    {"polynomial", KernelType::Polynomial},
    {"sigmoid", KernelType::Sigmoid}};
}  // namespace

using ado::core::Kernel;
using ado::core::KernelPolynomial;
using ado::core::KernelRBF;
using ado::core::KernelSigmoid;

SVM::SVM(const Float C, const Float tol, const std::string &kernel_type,
         const std::size_t max_steps, const std::size_t seed, const Float gamma,
         const Float degree, const Float coeff) {
  const auto kernel_item = KERNEL_MAP.find(kernel_type);
  if (kernel_item == KERNEL_MAP.end()) {
    throw std::runtime_error("Invalid kernel.");
  }

  std::unique_ptr<Kernel> kernel = nullptr;

  switch (kernel_item->second) {
    case KernelType::Polynomial: {
      kernel = std::make_unique<KernelPolynomial>(degree, gamma, coeff);
      break;
    }
    case KernelType::RBF: {
      kernel = std::make_unique<KernelRBF>(gamma);
      break;
    }
    case KernelType::Sigmoid:
    default: {
      kernel = std::make_unique<KernelSigmoid>(gamma, coeff);
      break;
    }
  }

  this->_svm = ado::core::SVM(C, tol, std::move(kernel), max_steps, seed);
};

void SVM::fit(xt::pyarray<double> &x, xt::pyarray<double> &y) {
  return this->_svm.fit(x, y);
}

xt::pyarray<double> SVM::fit_predict(xt::pyarray<double> &x,
                                     xt::pyarray<double> &y) {
  return this->_svm.fit_predict(x, y);
}

xt::pyarray<double> SVM::predict(xt::pyarray<double> &x) {
  return this->_svm.predict(x);
}

xt::pyarray<double> SVM::decision_function(xt::pyarray<double> &x) {
  return this->_svm.decision_function(x);
}