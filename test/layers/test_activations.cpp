#include <gtest/gtest.h>

#include <memory>

#include "ado/graph/variable.h"
#include "ado/layers/activations.h"
#include "ado/types.h"

using ado::Float;
using ado::FloatTensor;
using ado::FloatTensorShape;
using ado::graph::Variable;
using ado::layers::activations::Softmax;
using ado::math::dot;
using ado::math::Operand;
using ado::math::tr;

class TestActivations : public ::testing::Test {
 public:
  TestActivations() {}

  void SetUp() {
    // Define an input variable having shape 2-by-3 and gradient enabled.
    x_ = std::make_shared<Variable<Float>>(
        FloatTensor({{1000, 2000, 3000}, {4, 5, 6}}), false);

    w_ = std::make_shared<Variable<Float>>(
        FloatTensor({{0.1, 0.2, 0.3}, {0.5, 0.6, 0.7}}), true);
  }

 protected:
  std::shared_ptr<Variable<Float>> x_;
  std::shared_ptr<Variable<Float>> w_;
};

// Softmax activation.

TEST_F(TestActivations, SoftmaxForward) {
  auto softmax_op = Softmax<Float>();
  auto y = dot(this->x_, tr(this->w_));
  auto result = softmax_op(y);

  auto grad = FloatTensor(xt::ones<Float>({2, 2}));

  std::cout << result->forward() << std::endl;

  std::cout << "AAAAAAAAAAAA" << std::endl;

  result->backward(grad);
  std::cout << this->w_->grad() << std::endl;
}
