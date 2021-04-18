#include "ado/core/linear_svm.h"

#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

#include "ado/graph/variable.h"
#include "ado/layers/essentials.h"
#include "ado/losses/loss.h"
#include "ado/optimizers/optimizer.h"
#include "ado/utils/logger.h"
#include "ado/zoo/linear_svm_model.h"

namespace {
auto& logger = ado::utils::Logger::get();
}  // namespace

namespace ado {
namespace core {

using ado::graph::FloatVariable;
using ado::losses::HingeLoss;
using ado::optimizers::SGD;
using ado::utils::LogLevel;
using ado::zoo::LinearSVMModel;

LinearSVM::LinearSVM(const Float C, const Float lr, const Float momentum,
                     const Float decay, const Int epochs, const bool intercept,
                     const std::size_t seed)
    : C_(C),
      lr_(lr),
      momentum_(momentum),
      decay_(decay),
      epochs_(epochs),
      intercept_(intercept) {
  ado::random::seed(seed);
}

void LinearSVM::fit(const FloatTensor& x, const FloatTensor& y) {
  // Check target vector shape.
  auto y_target = y;
  if (y_target.shape().size() == 1) {
    y_target.reshape({y_target.shape(0), 1});
  }

  // Instantiate the linear SVM model.
  const std::size_t n_features = x.shape(1);
  this->_model =
      std::make_unique<LinearSVMModel<Float>>(n_features, this->intercept_);

  // Create the input variables.
  auto in_x = std::make_shared<FloatVariable>(x, false);
  auto in_y = std::make_shared<FloatVariable>(y_target, false);

  // Instantiate the loss function.
  auto loss = HingeLoss<Float>(this->C_);

  // Instantiate the optimizer.
  auto optimizer = SGD<Float>(this->_model->parameters(), this->lr_,
                              this->momentum_, this->decay_);

  // Training loop.
  for (auto epoch = 0; epoch < this->epochs_; ++epoch) {
    logger << LogLevel::Debug << "Epoch #" << epoch;

    // Zero the parameter gradients.
    optimizer.zero_grad();

    // Predict.
    auto pred = this->_model->operator()(in_x);

    // Compute the loss and backpropagate.
    auto loss_value = loss(pred, in_y);

    logger << LogLevel::Debug << "Loss: " << loss_value->forward();

    loss_value->backward();
    optimizer.step();
  }

  logger << LogLevel::Debug << "Training done !";
}

FloatTensor LinearSVM::fit_predict(const FloatTensor& x, const FloatTensor& y) {
  this->fit(x, y);
  return this->predict(x);
}

FloatTensor LinearSVM::predict(const FloatTensor& x) {
  auto y_hat = this->decision_function(x);
  filtration(y_hat, y_hat < 0) = -1;
  filtration(y_hat, y_hat > 0) = 1;
  return y_hat;
}

FloatTensor LinearSVM::decision_function(const FloatTensor& x) {
  auto in_x = std::make_shared<FloatVariable>(x, false);
  auto pred = this->_model->operator()(in_x);
  return xt::col(pred->forward(), 0);
}

}  // namespace core
}  // namespace ado