#ifndef ADO_BINDINGS_PY_SVM
#define ADO_BINDINGS_PY_SVM

#include <memory>
#include <string>

#include "ado/core/svm.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

using ado::Float;

struct SVM {
  SVM(const Float C, const Float tol, const std::string &kernel_type,
      const std::size_t max_steps, const std::size_t seed, const Float sigma,
      const Float degree, const Float alpha, const Float bias);

  void fit(xt::pyarray<double> &x, xt::pyarray<double> &y);
  xt::pyarray<double> fit_predict(xt::pyarray<double> &x,
                                  xt::pyarray<double> &y);

  xt::pyarray<double> predict(xt::pyarray<double> &x);
  xt::pyarray<double> decision_function(xt::pyarray<double> &x);

  ado::core::SVM _svm;
};

#endif  // ADO_BINDINGS_PY_SVM