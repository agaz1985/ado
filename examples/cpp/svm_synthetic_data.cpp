#include <iomanip>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "ado/core/kernel.h"
#include "ado/core/svm.h"
#include "ado/types.h"
#include "ado/utils/io.h"
#include "ado/utils/logger.h"

using ado::FloatArray;
using ado::core::Kernel;
using ado::core::KernelLinear;
using ado::core::KernelRBF;
using ado::core::SVM;
using ado::utils::load_data;
using ado::utils::LogFileHandler;
using ado::utils::Logger;
using ado::utils::LogLevel;
using ado::utils::LogStreamHandler;

FloatArray column_wise_normalization(const FloatArray& x) {
  auto c_max = xt::amax(x, 0);
  auto c_min = xt::amin(x, 0);
  return xt::eval((x - c_min) / (c_max - c_min));
}

void target_preprocessing(FloatArray& y) {
  filtration(y, xt::equal(y, 0)) = -1;
}

int main(int argc, char* argv[]) {
  // Define the logger and register the file, standard output and error
  // handlers.
  auto stream_handler = std::make_unique<LogStreamHandler>(LogLevel::Info);
  auto file_handler = std::make_unique<LogFileHandler>(
      "./log_cout.log", "./log_cerr.log", LogLevel::Debug);

  auto& logger = Logger::get();
  logger.register_handler(std::move(stream_handler));
  logger.register_handler(std::move(file_handler));

  // Define the random seed.
  const auto seed = 16;
  xt::random::seed(seed);

  // Define number of training and testing samples.
  const auto n_train_samples = 100;
  const auto n_test_samples = 30;

  // Load and shuffle the training data.
  logger << LogLevel::Info << "Loading training data...";

  FloatArray training_data = load_data("../data/occupancy/datatraining.csv");
  xt::random::shuffle(training_data);
  FloatArray x_train =
      xt::view(training_data, xt::range(0, n_train_samples), xt::range(0, 5));
  FloatArray y_train =
      xt::view(training_data, xt::range(0, n_train_samples), 5);

  x_train = column_wise_normalization(x_train);
  target_preprocessing(y_train);

  // Load and shuffle the testing data.
  logger << LogLevel::Info << "Loading test data...";

  FloatArray test_data = load_data("../data/occupancy/datatest2.csv");
  xt::random::shuffle(test_data);
  FloatArray x_test =
      xt::view(test_data, xt::range(0, n_test_samples), xt::range(0, 5));
  FloatArray y_test = xt::view(test_data, xt::range(0, n_test_samples), 5);

  x_test = column_wise_normalization(x_test);
  target_preprocessing(y_test);

  // Define the kernel.
  auto kernel = std::make_unique<KernelLinear>();

  logger << LogLevel::Info << "Fitting the SVM model...";
  auto svm = SVM(1.0, 1e-4, std::move(kernel), 100, seed);
  svm.fit(x_train, y_train);

  logger << LogLevel::Info << "Running inference on test data...";
  auto y_hat = svm.predict(x_test);

  // Computing the accuracy on the test set.
  const auto accuracy =
      xt::count_nonzero(xt::cast<uint8_t>(xt::equal(y_hat, y_test))) /
      (1.f * n_test_samples);

  logger << LogLevel::Info << std::setprecision(2)
         << "Accuracy: " << accuracy * 100 << " %";
}