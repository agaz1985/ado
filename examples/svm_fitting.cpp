#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "ado/kernel.h"
#include "ado/svm.h"
#include "ado/types.h"

using ado::Float;
using ado::FloatArray;
using ado::Kernel;
using ado::KernelLinear;
using ado::KernelRBF;
using ado::SVM;

FloatArray load_data(const std::string& filepath) {
  std::ifstream in_file;
  in_file.open(filepath);
  return xt::load_csv<Float>(in_file);
}

void save_data(const FloatArray& data, const std::string& filepath) {
  std::ofstream out_file;
  out_file.open(filepath);
  xt::dump_csv(out_file, data);
}

int main(int argc, char* argv[]) {
  FloatArray X = load_data("./data/x_data.csv");
  FloatArray y =
      xt::col(load_data("../data/y_data.csv"), 0);  // TODO: add to fit.

  std::array<std::unique_ptr<Kernel>, 2> kernel_list = {
      std::make_unique<KernelLinear>(), std::make_unique<KernelRBF>(5.0)};
  const auto n_samples = 40;

  // Generate data.
  const auto margin = 2;
  const auto k = xt::linspace<Float>(xt::amin(X)(0) - margin,
                                     xt::amax(X)(0) + margin, n_samples);
  auto k_mesh = xt::meshgrid(k, k);
  auto xx = std::get<0>(k_mesh);
  auto yy = std::get<1>(k_mesh);

  const FloatArray K = xt::stack(xtuple(xt::flatten(xx), xt::flatten(yy)), 1);

  for (auto kernel = kernel_list.begin(); kernel != kernel_list.end();
       ++kernel) {
    const auto kernel_type = (*kernel)->type();
    auto svm = SVM(1.0, 1e-4, std::move(*kernel), 1e3, 16);
    svm.fit(X, y);
    FloatArray M = svm.predict(K);
    M = M.reshape({n_samples, n_samples});

    save_data(M, "./data/m_data_" +
                     std::to_string(static_cast<int>(kernel_type)) + ".csv");
  }

  std::cout << "Done !" << std::endl;
}