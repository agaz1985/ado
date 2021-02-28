#include "pybind11/pybind11.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#define FORCE_IMPORT_ARRAY
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "ado/svm.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

namespace py = pybind11;

using ado::Float;
using ado::Kernel;
using ado::KernelLinear;
using ado::KernelRBF;
using ado::SVM;

struct svm {
  svm(const Float C = 1.0, const Float tol = 1e-4,
      const std::string &kernel_type = "linear",
      const std::size_t max_steps = 1e3, const std::size_t seed = 16,
      const Float sigma = 5.0) {
    if (kernel_type == "linear") {
      this->_svm = SVM(C, tol, std::move(std::make_unique<KernelLinear>()),
                       max_steps, seed);
    } else {
      this->_svm = SVM(C, tol, std::move(std::make_unique<KernelRBF>(sigma)),
                       max_steps, seed);
    }
  };

  void fit(xt::pyarray<double> &x, xt::pyarray<double> &y) {
    return this->_svm.fit(x, y);
  }

  xt::pyarray<double> fit_predict(xt::pyarray<double> &x,
                                  xt::pyarray<double> &y) {
    return this->_svm.fit_predict(x, y);
  }

  xt::pyarray<double> predict(xt::pyarray<double> &x) {
    return this->_svm.predict(x);
  }

  SVM _svm;
};

// Python Module and Docstrings

PYBIND11_MODULE(ado, m) {
  xt::import_numpy();

  m.doc() = R"pbdoc(
        ry

        .. currentmodule:: ado

        .. autosummary::
           :toctree: _generate

           svm
    )pbdoc";

  py::class_<svm>(m, "svm")
      .def(py::init<const Float, const Float, const std::string &,
                    const std::size_t, const std::size_t,
                    const Float>())  // TODO: give names to params and defaults.
      .def("fit", &svm::fit)
      .def("fit_predict", &svm::fit_predict)
      .def("predict", &svm::predict);
}
