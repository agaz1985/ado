#ifndef ADO_LOSSES_LOSS_H
#define ADO_LOSSES_LOSS_H

#include <limits.h>

#include "ado/constants.h"
#include "ado/graph/operator.h"
#include "ado/layers/essentials.h"
#include "ado/math/functions.h"
#include "ado/types.h"

namespace ado {
namespace losses {

using ado::graph::Operand;

using ado::math::clamp;
using ado::math::log;
using ado::math::mean;
using ado::math::pow;
using ado::math::sum;
using ado::math::operator+;
using ado::math::operator-;
using ado::math::operator*;
using ado::math::maximum;

using ado::EPS_FLOAT_VALUE;
using ado::layers::essentials::Linear;

template <typename T>
class Loss {
 public:
  enum class ReduceType { Mean = 0, Sum = 1 };

  Loss(const Loss<T>::ReduceType reduce) : reduce_(reduce) {}

  Operand<T> operator()(Operand<T> input, Operand<T> target) {
    auto loss = this->forward(input, target);

    switch (this->reduce_) {
      case Loss<T>::ReduceType::Mean:
        return mean(loss);
      case Loss<T>::ReduceType::Sum:
        return sum(loss);
      default:
        return mean(loss);
    }
  }

 protected:
  virtual Operand<T> forward(Operand<T> input, Operand<T> target) = 0;

 private:
  ReduceType reduce_;
};

template <typename T>
class BCELoss : public Loss<T> {
 public:
  BCELoss(const typename Loss<T>::ReduceType reduce = Loss<T>::ReduceType::Mean)
      : Loss<T>(reduce) {}

 protected:
  virtual Operand<T> forward(Operand<T> input, Operand<T> target) override {
    return -1 * (target * log(input + EPS_FLOAT_VALUE) +
                 (1 - target) * log(1 - input + EPS_FLOAT_VALUE));
  }
};

template <typename T>
class MSELoss : public Loss<T> {
 public:
  MSELoss(const typename Loss<T>::ReduceType reduce = Loss<T>::ReduceType::Mean)
      : Loss<T>(reduce) {}

 protected:
  virtual Operand<T> forward(Operand<T> input, Operand<T> target) override {
    return pow((input - target), 2.0);
  }
};

template <typename T>
class HingeLoss : public Loss<T> {
 public:
  HingeLoss(const Float C = 1.0, const typename Loss<T>::ReduceType reduce =
                                     Loss<T>::ReduceType::Mean)
      : C_(C), Loss<T>(reduce) {}

 protected:
  virtual Operand<T> forward(Operand<T> input, Operand<T> target) override {
    return this->C_ * maximum(0.0, 1.0 - target * input);
  }

 private:
  Float C_ = 1.0;
};

// 1 - target_pred * target_true

}  // namespace losses
}  // namespace ado

#endif  // ADO_LOSSES_LOSS_H
