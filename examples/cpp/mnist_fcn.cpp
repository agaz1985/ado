#include <iomanip>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include "ado/losses/loss.h"
#include "ado/optimizers/optimizer.h"
#include "ado/random/seed.h"
#include "ado/types.h"
#include "ado/utils/io.h"
#include "ado/utils/logger.h"
#include "ado/zoo/mnist_fc_model.h"

using ado::Float;
using ado::FloatTensor;
using ado::graph::FloatVariable;
using ado::losses::CrossEntropyLoss;
using ado::optimizers::SGD;
using ado::utils::load_data;
using ado::utils::LogFileHandler;
using ado::utils::Logger;
using ado::utils::LogLevel;
using ado::utils::LogStreamHandler;
using ado::zoo::MNISTFCModel;

namespace {
FloatTensor normalize_zero_one_range(const FloatTensor& x) {
  return xt::eval(x / 255.f);
}

FloatTensor to_one_hot(const FloatTensor& t, const std::size_t n_classes = 10) {
  FloatTensor one_hot = xt::zeros<Float>({t.shape(0), n_classes});
  one_hot[t] = 1;
  return one_hot;
}

FloatTensor to_categorical(const FloatTensor& t) { return xt::argmax(t, 1); }

}  // namespace

int main(int argv, char* argc[]) {
  // Define the logger and register the file, standard output and error
  // handlers.
  auto stream_handler = std::make_unique<LogStreamHandler>(LogLevel::Debug);
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

  FloatTensor training_data = load_data("../data/mnist/mnist_train.csv");
  xt::random::shuffle(training_data);
  FloatTensor x_train =
      xt::view(training_data, xt::range(0, 100), xt::range(1, 785));
  FloatTensor y_train = xt::view(training_data, xt::range(0, 100), 0); //TODO: fix sum axis backprop and add keep axes option.

  // Normalize each image in the training set to 0-1 range.
  x_train = normalize_zero_one_range(x_train);
  y_train = to_one_hot(y_train);

  // Load and shuffle the testing data.
  logger << LogLevel::Info << "Loading test data...";

  FloatTensor test_data = load_data("../data/mnist/mnist_test.csv");
  xt::random::shuffle(test_data);
  FloatTensor x_test = xt::view(test_data, xt::all(), xt::range(1, 785));
  FloatTensor y_test = xt::view(test_data, xt::all(), 0);

  // Normalize each image in the training set to 0-1 range.
  x_test = normalize_zero_one_range(x_test);

  // TODO: move this.
  if (y_train.shape().size() == 1) {
    y_train.reshape({y_train.shape(0), 1});
  }

  // Define the input data.
  auto x_train_data = std::make_shared<FloatVariable>(x_train, false);
  auto y_train_data = std::make_shared<FloatVariable>(y_train, false);

  // Instantiate the MNIST model.
  auto model = MNISTFCModel<Float>(784, 32, 10);

  // Instantiate the loss function.
  auto loss = CrossEntropyLoss<Float>();

  // Instantiate the optimizer.
  auto lr = 1e-3;  // learning rate.
  auto optimizer = SGD<Float>(model.parameters(), lr, 0.99, 0.0);

  // Training loop.
  auto epochs = 5;
  for (auto epoch = 0; epoch < epochs; ++epoch) {
    logger << LogLevel::Debug << "Epoch #" << epoch;

    // Zero the parameter gradients.
    optimizer.zero_grad();

    // Predict.
    auto pred = model(x_train_data);

    // Compute the loss and backpropagate.
    auto loss_value = loss(pred, y_train_data);

    logger << LogLevel::Debug << "Loss: " << loss_value->forward();

    loss_value->backward();
    optimizer.step();
  }

  logger << LogLevel::Info << "Training done !";

  logger << LogLevel::Info << "Running inference on test data...";

  // Define the input data.
  auto x_test_data = std::make_shared<FloatVariable>(x_test, false);

  auto y_hat_one_hot = model(x_test_data);
  auto y_hat = to_categorical(y_hat_one_hot->forward());

  // Computing the accuracy on the test set.
  const auto accuracy =
      xt::count_nonzero(xt::cast<uint8_t>(xt::equal(y_hat, y_test))) /
      (1.f * y_test.size());

  logger << LogLevel::Info << std::setprecision(2)
         << "Accuracy: " << accuracy * 100 << " %";

  return 0;
}