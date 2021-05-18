#include <gtest/gtest.h>

#include <memory>

#include "ado/graph/variable.h"
#include "ado/math/unary_operators.h"
#include "ado/types.h"

using ado::Float;
using ado::FloatTensor;
using ado::FloatTensorShape;
using ado::graph::Variable;
using ado::math::Operand;
using ado::math::SumOperator;

class TestUnaryOperators : public ::testing::Test {
 public:
  TestUnaryOperators() {}

  void SetUp() {
    // Define an input variable having shape 2-by-3 and gradient enabled.
    x_ = std::make_shared<Variable<Float>>(FloatTensor({{1, 2, 3}, {4, 5, 6}}),
                                           true);
  }

 protected:
  std::shared_ptr<Variable<Float>> x_;
};

// Sum operator.

TEST_F(TestUnaryOperators, SumOperatorForward) {
  // Sum over all dimensions by not specifing any axis.
  auto sum_op = SumOperator<Float>(this->x_);

  auto result = sum_op.forward();
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(21, result(0));

  // Sum over all dimensions by specifing an empty axes vector.
  sum_op = SumOperator<Float>(this->x_, {});

  result = sum_op.forward();
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(21, result(0));

  // Sum over the first dimension without keeping the reduced dimension.
  sum_op = SumOperator<Float>(this->x_, {0}, false);
  result = sum_op.forward();

  EXPECT_EQ(3, result.size());
  EXPECT_EQ(FloatTensorShape({3}), result.shape());
  EXPECT_EQ(FloatTensor({5, 7, 9}), result);

  // Sum over the first dimension by keeping the reduced dimension.
  sum_op = SumOperator<Float>(this->x_, {0}, true);
  result = sum_op.forward();
  auto expected_result = FloatTensor({5, 7, 9});
  expected_result = expected_result.reshape({1, 3});

  EXPECT_EQ(3, result.size());
  EXPECT_EQ(FloatTensorShape({1, 3}), result.shape());
  EXPECT_EQ(expected_result, result);

  // Sum over the second dimension without keeping the reduced dimension.
  sum_op = SumOperator<Float>(this->x_, {1}, false);
  result = sum_op.forward();

  EXPECT_EQ(2, result.size());
  EXPECT_EQ(FloatTensorShape({2}), result.shape());
  EXPECT_EQ(FloatTensor({6, 15}), result);

  // Sum over the first dimension by keeping the reduced dimension.
  sum_op = SumOperator<Float>(this->x_, {1}, true);
  expected_result = FloatTensor({6, 15});
  expected_result = expected_result.reshape({2, 1});
  result = sum_op.forward();

  EXPECT_EQ(2, result.size());
  EXPECT_EQ(FloatTensorShape({2, 1}), result.shape());
  EXPECT_EQ(expected_result, result);
}

TEST_F(TestUnaryOperators, SumOperatorBackward) {
  // Define the gradient variable.
  auto grad = FloatTensor({5});
  auto expected_grad = FloatTensor({{5, 5, 5}, {5, 5, 5}});

  // Sum over all dimensions by not specifing any axis.
  auto sum_op = SumOperator<Float>(this->x_);

  sum_op.backward(grad);
  auto computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);

  // Sum over all dimensions by specifing an empty axes vector.
  this->x_->zero_grad();  // Reset the gradient.
  sum_op = SumOperator<Float>(this->x_, {});

  sum_op.backward(grad);
  computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);

  // Sum over the first dimension without keeping the reduced dimension.
  this->x_->zero_grad();  // Reset the gradient.
  sum_op = SumOperator<Float>(this->x_, {0}, false);

  sum_op.backward(grad);
  computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);

  // Sum over the first dimension by keeping the reduced dimension.
  this->x_->zero_grad();  // Reset the gradient.
  sum_op = SumOperator<Float>(this->x_, {0}, true);

  sum_op.backward(grad);
  computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);

  // Sum over the second dimension without keeping the reduced dimension.
  this->x_->zero_grad();  // Reset the gradient.
  sum_op = SumOperator<Float>(this->x_, {1}, false);

  sum_op.backward(grad);
  computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);

  // Sum over the second dimension by keeping the reduced dimension.
  this->x_->zero_grad();  // Reset the gradient.
  sum_op = SumOperator<Float>(this->x_, {1}, true);

  sum_op.backward(grad);
  computed_grad = this->x_->grad();

  EXPECT_EQ(this->x_->shape(), computed_grad.shape());
  EXPECT_EQ(expected_grad, computed_grad);
}
