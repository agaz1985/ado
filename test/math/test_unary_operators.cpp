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

  void SetUp() { x_ = FloatTensor({{1, 2, 3}, {4, 5, 6}}); }

  void TearDown() {}

 protected:
  FloatTensor x_;
};

// Sum operator.

TEST_F(TestUnaryOperators, SumOperatorForward) {
  // Define an input variable having shape 2-by-3.
  auto x = std::make_shared<Variable<Float>>(this->x_);

  // Sum over all dimensions by not specifing any axis.
  auto sum_op = SumOperator<Float>(x);

  auto result = sum_op.forward();
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(21, result(0));

  // Sum over all dimensions by specifing an empty axes vector.
  sum_op = SumOperator<Float>(x, {});

  result = sum_op.forward();
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(21, result(0));

  // Sum over the first dimension without keeping the reduced dimension.
  sum_op = SumOperator<Float>(x, {0}, false);
  result = sum_op.forward();

  EXPECT_EQ(3, result.size());
  EXPECT_EQ(FloatTensorShape({3}), result.shape());
  EXPECT_EQ(FloatTensor({5, 7, 9}), result);

  // Sum over the first dimension by keeping the reduced dimension.
  sum_op = SumOperator<Float>(x, {0}, true);
  result = sum_op.forward();
  auto expected_result = FloatTensor({5, 7, 9});
  expected_result = expected_result.reshape({1, 3});

  EXPECT_EQ(3, result.size());
  EXPECT_EQ(FloatTensorShape({1, 3}), result.shape());
  EXPECT_EQ(expected_result, result);

  // Sum over the second dimension without keeping the reduced dimension.
  sum_op = SumOperator<Float>(x, {1}, false);
  result = sum_op.forward();

  EXPECT_EQ(2, result.size());
  EXPECT_EQ(FloatTensorShape({2}), result.shape());
  EXPECT_EQ(FloatTensor({6, 15}), result);

  // Sum over the first dimension by keeping the reduced dimension.
  sum_op = SumOperator<Float>(x, {1}, true);
  expected_result = FloatTensor({6, 15});
  expected_result = expected_result.reshape({2, 1});
  result = sum_op.forward();

  EXPECT_EQ(2, result.size());
  EXPECT_EQ(FloatTensorShape({2, 1}), result.shape());
  EXPECT_EQ(expected_result, result);
}

TEST_F(TestUnaryOperators, SumOperatorBackward) { EXPECT_TRUE(true); }
