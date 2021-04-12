#include "ado/core/svm.h"

#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

#include "ado/utils/logger.h"

namespace {
auto& logger = ado::utils::Logger::get();

using ado::Float;

Float clip_value(const Float value, const Float high, const Float low) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

}  // namespace

namespace ado {
namespace core {

using ado::utils::LogLevel;

SVM::SVM(const Float C, const Float tol, std::unique_ptr<Kernel> kernel,
         const std::size_t max_steps, const std::size_t seed)
    : C_(C),
      tol_(tol),
      kernel_(std::move(kernel)),
      max_steps_(max_steps) {
  xt::random::seed(seed);
}

void SVM::fit(const FloatTensor& x, const FloatTensor& y) {
  // Check target vector shape.
  auto y_target = y;
  if (y_target.shape().size() == 2) {
    y_target = xt::col(y, 0);
  }

  const std::size_t n_samples = x.shape(0);
  this->alphas_ = xt::zeros<Float>({n_samples});
  this->errors_ = xt::zeros<Float>({n_samples});

  std::size_t num_changed = 0;
  bool examine_all = true;
  std::size_t remaining_steps = this->max_steps_;

  logger << LogLevel::Info << "Fitting " << n_samples
         << " samples for a maximum of " << this->max_steps_ << " steps.";

  while ((num_changed > 0 || examine_all) && (remaining_steps > 0)) {
    --remaining_steps;

    logger << LogLevel::Debug << "Remaining steps: " << remaining_steps;

    num_changed = 0;
    if (examine_all) {
      for (std::size_t idx = 0; idx < n_samples; ++idx) {
        num_changed += this->examine_example(idx, x, y_target);
      }
    } else {
      const auto condition = ((this->alphas_ < this->tol_) ||
                              (this->alphas_ > (this->C_ - this->tol_)));
      const SizeArray filtered_indexes =
          xt::flatten_indices(xt::where(condition));

      for (std::size_t idx : filtered_indexes) {
        num_changed += this->examine_example(idx, x, y_target);
      }
    }

    if (examine_all)
      examine_all = false;
    else if (num_changed == 0)
      examine_all = true;
  }

  auto filtered_idxs = xt::flatten_indices(xt::nonzero(this->alphas_));
  this->x_support_ = xt::view(x, xt::keep(filtered_idxs), xt::all());
  this->y_support_ = xt::filter(y_target, xt::not_equal(this->alphas_, 0));
  this->alphas_ = xt::filter(this->alphas_, xt::not_equal(this->alphas_, 0));
}

FloatTensor SVM::fit_predict(const FloatTensor& x, const FloatTensor& y) {
  this->fit(x, y);
  return this->predict(x);
}

FloatTensor SVM::predict(const FloatTensor& x) {
  auto y_hat = this->decision_function(x);
  filtration(y_hat, y_hat < 0) = -1;
  filtration(y_hat, y_hat > 0) = 1;
  return y_hat;
}

FloatTensor SVM::decision_function(const FloatTensor& x) {
  FloatTensor predictions = xt::zeros<Float>({x.shape(0)});

  for (std::size_t idx = 0; idx < x.shape(0); ++idx) {
    predictions[idx] = this->eval(this->x_support_, this->y_support_,
                                  this->alphas_, xt::view(x, idx, xt::all()));
  }
  return predictions;
}

std::int8_t SVM::examine_example(const std::size_t i2, const FloatTensor& x,
                                 const FloatTensor& y) {
  const auto y2 = y(i2);
  const auto alph2 = this->alphas_[i2];

  auto e2 = this->errors_[i2];

  if ((alph2 < this->tol_) || alph2 > (this->C_ - this->tol_)) {
    auto filtered_idxs = xt::flatten_indices(xt::nonzero(this->alphas_));
    FloatTensor x_filtered = xt::view(x, xt::keep(filtered_idxs), xt::all());
    FloatTensor y_filtered = xt::filter(y, xt::not_equal(this->alphas_, 0));
    FloatTensor alphas_filtered =
        xt::filter(this->alphas_, xt::not_equal(this->alphas_, 0));
    e2 = this->eval(x_filtered, y_filtered, alphas_filtered,
                    xt::view(x, i2, xt::all())) -
         y2;
  }

  auto r2 = e2 * y2;
  if ((r2 < -this->tol_ && alph2 < this->C_) ||
      (r2 > this->tol_ && alph2 > 0)) {
    const auto condition = ((this->alphas_ < this->tol_) ||
                            (this->alphas_ > (this->C_ - this->tol_)));
    SizeArray filtered_indexes = xt::flatten_indices(xt::where(condition));

    if (filtered_indexes.size() > 0) {
      const auto i1 = xt::argmax(this->errors_);
      if (this->take_step(i1(0), i2, x, y, y2, alph2, e2)) {
        return 1;
      }
    }

    if (filtered_indexes.size() > 0) {
      xt::random::shuffle(filtered_indexes);
      for (auto idx : filtered_indexes) {
        if (this->take_step(idx, i2, x, y, y2, alph2, e2)) {
          return 1;
        }
      }
    }

    SizeArray all_indexes = xt::arange<std::size_t>(0, x.shape(0));
    xt::random::shuffle(all_indexes);
    for (auto idx : all_indexes) {
      if (this->take_step(idx, i2, x, y, y2, alph2, e2)) {
        return 1;
      }
    }
  }
  return 0;
}

Float SVM::eval(const FloatTensor& x, const FloatTensor& y,
                const FloatTensor& alphas, const FloatTensor& xi) const {
  if (y.size() == 0) {
    return -this->b_;
  }

  auto w_x = xt::sum(alphas * this->kernel_->operator()(x, xi) * y);
  return w_x(0) - this->b_;
}

Float SVM::compute_b(const Float& e1, const Float& e2, const Float& y1,
                     const Float& a1, const Float& alph1, const Float& y2,
                     const Float& a2, const Float& alph2, const Float& k11,
                     const Float& k12, const Float& k22) const {
  const auto b1 =
      e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + this->b_;
  const auto b2 =
      e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + this->b_;

  if ((a1 > 0) && (a1 < this->C_))
    return b1;
  else if ((a2 > 0) && (a2 < this->C_))
    return b2;
  else
    return (b1 + b2) / 2.0;
}

Float SVM::compute_gamma(const Float& alph1, const Float& alph2, const Float& V,
                         const Float& k11, const Float& k12, const Float& k22,
                         const Float& s, const Float& y1, const Float& y2,
                         const Float& e1, const Float& e2) const {
  const auto f1 = y1 * (e1 + this->b_) - alph1 * k11 - s * alph2 * k12;
  const auto f2 = y2 * (e2 + this->b_) - s * alph1 * k12 - alph2 * k22;
  const auto V1 = alph1 + s * (alph2 - V);
  return V1 * f1 + V * f2 + 0.5 * (V1 * V1) * k11 + 0.5 * (V * V) * k22 +
         s * V * V1 * k12;
}

Float SVM::kernel_function(const FloatTensor& x1, const FloatTensor& x2) const {
  return this->kernel_->operator()(x1, x2)(0);
}

std::int8_t SVM::take_step(const std::size_t i1, const std::size_t i2,
                           const FloatTensor& x, const FloatTensor& y,
                           const Float& y2, const Float& alph2,
                           const Float& e2) {
  if (i1 == i2) return 0;

  Float alph1 = this->alphas_[i1];
  Float y1 = y(i1);

  Float e1 = this->errors_[i1];
  if ((alph1 < this->tol_) || (alph1 > (this->C_ - this->tol_))) {
    auto filtered_idxs = xt::flatten_indices(xt::nonzero(this->alphas_));
    FloatTensor x_filtered = xt::view(x, xt::keep(filtered_idxs), xt::all());
    FloatTensor y_filtered = xt::filter(y, xt::not_equal(this->alphas_, 0));
    FloatTensor alphas_filtered =
        xt::filter(this->alphas_, xt::not_equal(this->alphas_, 0));
    e1 = this->eval(x_filtered, y_filtered, alphas_filtered,
                    xt::view(x, i1, xt::all())) -
         y1;
  }

  auto s = y1 * y2;

  Float L = 0;
  Float H = 0;

  if (y1 != y2) {
    L = std::max(Float(0.0), alph2 - alph1);
    H = std::min(this->C_, this->C_ + alph2 - alph1);
  } else {
    L = std::max(Float(0.0), alph2 + alph1 - this->C_);
    H = std::min(this->C_, alph2 + alph1);
  }

  if (L == H) return 0;

  const auto k11 = this->kernel_function(xt::view(x, i1, xt::all()),
                                         xt::view(x, i1, xt::all()));
  const auto k12 = this->kernel_function(xt::view(x, i1, xt::all()),
                                         xt::view(x, i2, xt::all()));
  const auto k22 = this->kernel_function(xt::view(x, i2, xt::all()),
                                         xt::view(x, i2, xt::all()));
  const auto eta = k11 + k22 - 2 * k12;

  Float a2 = 0.0;
  if (eta > 0) {
    a2 = clip_value(alph2 + y2 * (e1 - e2) / eta, H, L);
  } else {
    const auto Lobj =
        this->compute_gamma(alph1, alph2, L, k11, k12, k22, s, y1, y2, e1, e2);
    const auto Hobj =
        this->compute_gamma(alph1, alph2, H, k11, k12, k22, s, y1, y2, e1, e2);
    if (Lobj < (Hobj - this->tol_))
      a2 = L;
    else if (Lobj > (Hobj + this->tol_))
      a2 = H;
    else
      a2 = alph2;
  }

  if (std::abs(a2 - alph2) < this->tol_ * (a2 + alph2 + this->tol_)) {
    return 0;
  }

  const auto a1 = alph1 + s * (alph2 - a2);
  auto new_b =
      this->compute_b(e1, e2, y1, a1, alph1, y2, a2, alph2, k11, k12, k22);
  auto delta_b = new_b - this->b_;
  this->b_ = new_b;

  // Error cache.
  auto t1 = y1 * (a1 - alph1);
  auto t2 = y2 * (a2 - alph2);

  for (std::size_t idx = 0; idx < this->alphas_.size(); ++idx) {
    if ((this->alphas_[idx] > 0) && (this->alphas_[idx] < this->C_)) {
      this->errors_[idx] +=
          t1 * this->kernel_function(xt::view(x, i1, xt::all()),
                                     xt::view(x, idx, xt::all())) +
          t2 * this->kernel_function(xt::view(x, i2, xt::all()),
                                     xt::view(x, idx, xt::all())) -
          delta_b;
    }
  }

  this->errors_(i1) = 0.0;
  this->errors_(i2) = 0.0;

  this->alphas_(i1) = a1;
  this->alphas_(i2) = a2;

  return 1;
}

}  // namespace core
}  // namespace ado