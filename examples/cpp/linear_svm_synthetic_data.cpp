#include <iomanip>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "ado/core/linear_svm.h"
#include "ado/random/seed.h"
#include "ado/types.h"
#include "ado/utils/io.h"
#include "ado/utils/logger.h"

using ado::FloatTensor;
using ado::core::LinearSVM;
using ado::utils::load_data;
using ado::utils::LogFileHandler;
using ado::utils::Logger;
using ado::utils::LogLevel;
using ado::utils::LogStreamHandler;

namespace {
FloatTensor column_wise_normalization(const FloatTensor& x) {
  auto c_max = xt::amax(x, 0);
  auto c_min = xt::amin(x, 0);
  return xt::eval((x - c_min) / (c_max - c_min));
}

void target_preprocessing(FloatTensor& y) {
  filtration(y, xt::equal(y, 0)) = -1;
}
}  // namespace

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
  ado::random::seed(16);

  // Load and shuffle the training data.
  logger << LogLevel::Info << "Loading training data...";

  FloatTensor training_data = load_data("../data/occupancy/datatraining.csv");
  xt::random::shuffle(training_data);
  FloatTensor x_train = xt::view(training_data, xt::all(), xt::range(0, 5));
  FloatTensor y_train = xt::view(training_data, xt::all(), 5);

  x_train = column_wise_normalization(x_train);
  target_preprocessing(y_train);

  // Load and shuffle the testing data.
  logger << LogLevel::Info << "Loading test data...";

  FloatTensor test_data = load_data("../data/occupancy/datatest2.csv");
  xt::random::shuffle(test_data);
  FloatTensor x_test = xt::view(test_data, xt::all(), xt::range(0, 5));
  FloatTensor y_test = xt::view(test_data, xt::all(), 5);

  x_test = column_wise_normalization(x_test);
  target_preprocessing(y_test);

  logger << LogLevel::Info << "Fitting the linear SVM model on "
         << y_train.size() << " training samples...";

  auto linear_svm = LinearSVM(10.0, 1e-3, 0.99, 0.0, 1e3, true, seed);
  linear_svm.fit(x_train, y_train);

  logger << LogLevel::Info << "Running inference on " << y_test.size()
         << " test samples...";

  auto y_hat = linear_svm.predict(x_test);

  // Computing the accuracy on the test set.
  const auto accuracy =
      xt::count_nonzero(xt::cast<uint8_t>(xt::equal(y_hat, y_test))) /
      (1.f * y_test.size());

  logger << LogLevel::Info << std::setprecision(2)
         << "Accuracy: " << accuracy * 100 << " %";
}
