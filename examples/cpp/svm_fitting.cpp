#include <exception>
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
  std::ifstream in_file(filepath);
  if (in_file.good()) {
    return xt::load_csv<Float>(in_file);
  }
  throw std::runtime_error("File does not exist !");
}

void save_data(const FloatArray& data, const std::string& filepath) {
  std::ofstream out_file;
  out_file.open(filepath);
  xt::dump_csv(out_file, data);
}

int main(int argc, char* argv[]) {
  // Load training data.
  std::cout << "Loading training data..." << std::endl;
  FloatArray training_data = load_data("../data/occupancy/datatraining.csv");
  FloatArray x_train = xt::view(training_data, xt::all(), xt::range(0, 5));
  FloatArray y_train = xt::view(training_data, xt::all(), 5);

  // Load testing data.
  std::cout << "Loading test data..." << std::endl;
  FloatArray test_data = load_data("../data/occupancy/datatest2.csv");
  FloatArray x_test = xt::view(test_data, xt::all(), xt::range(0, 5));
  FloatArray y_test = xt::view(test_data, xt::all(), 5);

  // Define the kernel.
  auto kernel = std::make_unique<KernelLinear>();

  std::cout << "Fitting the SVM model..." << std::endl;
  auto svm = SVM(1.0, 1e-4, std::move(kernel), 1e3, 16);
  svm.fit(x_train, y_train);

  std::cout << "Run inference on the test set..." << std::endl;
  auto y_hat = svm.predict(x_test);

  std::cout << y_hat << std::endl;
}