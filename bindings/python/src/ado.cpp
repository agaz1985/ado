#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "svm.h"
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

// Python Module and Docstrings

PYBIND11_MODULE(ado, m) {
  xt::import_numpy();

  py::class_<SVM>(m, "SVM")
      .def(py::init<const Float, const Float, const std::string &,
                    const std::size_t, const std::size_t, const Float,
                    const Float, const Float>(),
           py::arg("C") = 1.0, py::arg("tol") = 1e-4,
           py::arg("kernel") = "linear", py::arg("max_steps") = 1000,
           py::arg("seed") = 16, py::arg("gamma") = 1.0,
           py::arg("degree") = 1.0, py::arg("coeff") = 0.0)
      .def("fit", &SVM::fit, "Fit the model on the input data.", py::arg("x"),
           py::arg("y"))
      .def("fit_predict", &SVM::fit_predict,
           "Fit the model and subsequently run inference on the input data.",
           py::arg("x"), py::arg("y"))
      .def("predict", &SVM::predict,
           "Run inference on the input data and return the predicted classes.",
           py::arg("x"))
      .def("decision_function", &SVM::decision_function,
           "Run inference on the input data and return the confidence score.",
           py::arg("x"));
}
