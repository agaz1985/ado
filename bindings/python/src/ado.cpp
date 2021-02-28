#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include "svm.h"

namespace py = pybind11;

// Python Module and Docstrings

PYBIND11_MODULE(ado, m) {
  xt::import_numpy();

  py::class_<SVM>(m, "SVM")
      .def(py::init<const Float, const Float, const std::string &,
                    const std::size_t, const std::size_t, const Float>(),
           py::arg("C") = 1.0, py::arg("tol") = 1e-4,
           py::arg("kernel") = "linear", py::arg("max_steps") = 1000,
           py::arg("seed") = 16, py::arg("sigma") = 5.0)
      .def("fit", &SVM::fit, "Fit the model on the input data", py::arg("x"),
           py::arg("y"))
      .def("fit_predict", &SVM::fit_predict,
           "Fit the model and subsequently run inference on the input data",
           py::arg("x"), py::arg("y"))
      .def("predict", &SVM::predict, "Run inference on the input data",
           py::arg("x"));
}
