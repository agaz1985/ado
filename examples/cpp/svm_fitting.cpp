#include <exception>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
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

// TODO:
// move set seed to library.
// move load/save to library.
// move normalize to library.
// clean up includes.
// ADD Reference to Data and disclaimer.
// Fix prob bindings.

FloatArray load_data(const std::string& filepath) {
  std::fstream s(filepath.c_str(), std::ios::in);
  if (!s) {
    throw std::runtime_error("File does not exist !");
  } else {
    return xt::load_csv<double>(s);
  }
}

void save_data(const FloatArray& data, const std::string& filepath) {
  std::ofstream out_file;
  out_file.open(filepath);
  xt::dump_csv(out_file, data);
}

FloatArray normalize_data(const FloatArray& x) {
  auto c_max = xt::amax(x, 0);
  auto c_min = xt::amin(x, 0);
  return xt::eval((x - c_min) / (c_max - c_min));
}

void preprocess_labels(FloatArray& y) { filtration(y, xt::equal(y, 0)) = -1; }

int main(int argc, char* argv[]) {
  // Define the random seed.
  const auto seed = 16;
  xt::random::seed(seed);  // TODO: move this to library.

  // Define number of training and testing samples.
  const auto n_train_samples = 100;
  const auto n_test_samples = 30;

  // Load and shuffle the training data.
  std::cout << "Loading training data..." << std::endl;
  FloatArray training_data = load_data("../data/occupancy/datatraining.csv");
  xt::random::shuffle(training_data);
  FloatArray x_train =
      xt::view(training_data, xt::range(0, n_train_samples), xt::range(0, 5));
  FloatArray y_train =
      xt::view(training_data, xt::range(0, n_train_samples), 5);

  x_train = normalize_data(x_train);
  preprocess_labels(y_train);

  // Load and shuffle the testing data.
  std::cout << "Loading test data..." << std::endl;
  FloatArray test_data = load_data("../data/occupancy/datatest2.csv");
  xt::random::shuffle(test_data);
  FloatArray x_test =
      xt::view(test_data, xt::range(0, n_test_samples), xt::range(0, 5));
  FloatArray y_test = xt::view(test_data, xt::range(0, n_test_samples), 5);

  x_test = normalize_data(x_test);
  preprocess_labels(y_test);

  // Define the kernel.
  auto kernel = std::make_unique<KernelLinear>();

  std::cout << "Fitting the SVM model..." << std::endl;
  auto svm = SVM(1.0, 1e-4, std::move(kernel), 100, seed);
  svm.fit(x_train, y_train);

  std::cout << "Run inference on the test set..." << std::endl;
  auto y_hat = svm.predict(x_test);

  // Computing the accuracy on the test set.
  const auto accuracy =
      xt::count_nonzero(xt::cast<uint8_t>(xt::equal(y_hat, y_test))) /
      (1.f * n_test_samples);
  std::cout << std::setprecision(2) << "Accuracy: " << accuracy * 100 << " %" << std::endl;
}