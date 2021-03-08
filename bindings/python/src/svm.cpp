#include "svm.h"

SVM::SVM(const Float C, const Float tol, const std::string &kernel_type,
         const std::size_t max_steps, const std::size_t seed,
         const Float sigma) {
  if (kernel_type == "linear") {
    this->_svm = ado::SVM(C, tol, std::move(std::make_unique<KernelLinear>()),
                          max_steps, seed);
  } else {
    this->_svm = ado::SVM(C, tol, std::move(std::make_unique<KernelRBF>(sigma)),
                          max_steps, seed);
  }
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

xt::pyarray<double> SVM::prob(xt::pyarray<double> &x) {
  return this->_svm.prob(x);
}