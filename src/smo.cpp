#include "ado/smo.h"

#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

SMO::SMO(const Float C, const Float tol, const KernelType kernel_type,
         const Float sigma, const std::size_t max_steps, const std::size_t seed)
    : _C(C),
      _tol(tol),
      _kernel_type(kernel_type),
      _sigma(sigma),
      _max_steps(max_steps),
      _seed(seed) {
  xt::random::seed(16);
}

void SMO::fit(const FloatArray& x, const FloatArray& y) {
  const std::size_t n_samples = x.shape(0);
  this->_alphas = xt::zeros<Float>({n_samples});
  this->_errors = xt::zeros<Float>({n_samples});

  std::size_t num_changed = 0;
  bool examine_all = true;
  std::size_t remaining_steps = this->_max_steps;

  while ((num_changed > 0 || examine_all) && (remaining_steps > 0)) {
    --remaining_steps;
    num_changed = 0;
    if (examine_all) {
      for (std::size_t idx = 0; idx < n_samples; ++idx) {
        num_changed += this->examine_example(idx, x, y);
      }
    } else {
      const auto condition = ((this->_alphas < this->_tol) ||
                              (this->_alphas > (this->_C - this->_tol)));
      const SizeArray filtered_indexes =
          xt::flatten_indices(xt::where(condition));

      for (std::size_t idx : filtered_indexes) {
        num_changed += this->examine_example(idx, x, y);
      }
    }

    if (examine_all)
      examine_all = false;
    else if (num_changed == 0)
      examine_all = true;
  }

  auto filtered_idxs =
      xt::flatten_indices(xt::argwhere(xt::not_equal(this->_alphas, 0)));
  this->_x_support = xt::view(x, xt::keep(filtered_idxs), xt::all());
  this->_y_support = xt::filter(y, xt::not_equal(this->_alphas, 0));
  this->_alphas = xt::filter(this->_alphas, xt::not_equal(this->_alphas, 0));
}

FloatArray SMO::fit_predict(const FloatArray& x, const FloatArray& y) {
  this->fit(x, y);
  return this->predict(x);
}

FloatArray SMO::predict(const FloatArray& x) {
  FloatArray predictions = xt::zeros<Float>({x.shape(0)});

  for (std::size_t idx = 0; idx < x.shape(0); ++idx) {
    predictions[idx] = this->eval(this->_x_support, this->_y_support,
                                  this->_alphas, xt::view(x, idx, xt::all()));
  }
  return predictions;
}

std::int8_t SMO::examine_example(const std::size_t i2, const FloatArray& x,
                                 const FloatArray& y) {
  const auto y2 = y(i2);
  const auto alph2 = this->_alphas[i2];

  auto e2 = this->_errors[i2];
  if ((alph2 < this->_tol) || alph2 > (this->_C - this->_tol)) {
    auto filtered_idxs =
        xt::flatten_indices(xt::argwhere(xt::not_equal(this->_alphas, 0)));
    FloatArray x_filtered = xt::view(x, xt::keep(filtered_idxs), xt::all());
    FloatArray y_filtered = xt::filter(y, xt::not_equal(this->_alphas, 0));
    FloatArray alphas_filtered =
        xt::filter(this->_alphas, xt::not_equal(this->_alphas, 0));
    e2 = this->eval(x_filtered, y_filtered, alphas_filtered,
                    xt::view(x, i2, xt::all())) -
         y2;
  }

  auto r2 = e2 * y2;
  if ((r2 < -this->_tol && alph2 < this->_C) ||
      (r2 > this->_tol && alph2 > 0)) {
    const auto condition = ((this->_alphas < this->_tol) ||
                            (this->_alphas > (this->_C - this->_tol)));
    SizeArray filtered_indexes = xt::flatten_indices(xt::where(condition));

    if (filtered_indexes.size() > 0) {
      const auto i1 = xt::argmax(this->_errors);
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

Float SMO::eval(const FloatArray& x, const FloatArray& y,
                const FloatArray& alphas, const FloatArray& xi) const {
  Float w_x = 0.0;
  for (std::size_t idx = 0; idx < y.size(); ++idx) {
    w_x += alphas[idx] * y(idx) *
           this->kernel_function(xt::view(x, idx, xt::all()), xi);
  }
  return w_x - this->_b;
}

Float SMO::compute_b(const Float& e1, const Float& e2, const Float& y1,
                     const Float& a1, const Float& alph1, const Float& y2,
                     const Float& a2, const Float& alph2, const Float& k11,
                     const Float& k12, const Float& k22) const {
  const auto b1 =
      e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + this->_b;
  const auto b2 =
      e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + this->_b;

  if ((a1 > 0) && (a1 < this->_C))
    return b1;
  else if ((a2 > 0) && (a2 < this->_C))
    return b2;
  else
    return (b1 + b2) / 2.0;
}

FloatArray SMO::compute_w(const FloatArray& x1, const FloatArray& x2,
                          const FloatArray& y1, const FloatArray& y2,
                          const FloatArray& a1, const FloatArray& a2,
                          const FloatArray& alph1,
                          const FloatArray& alph2) const {
  return this->_w + y1 * (a1 - alph1) * x1 + y2 * (a2 - alph2) * x2;
}

Float SMO::compute_gamma(const Float& alph1, const Float& alph2, const Float& V,
                         const Float& k11, const Float& k12, const Float& k22,
                         const Float& s, const Float& y1, const Float& y2,
                         const Float& e1, const Float& e2) const {
  const auto f1 = y1 * (e1 + this->_b) - alph1 * k11 - s * alph2 * k12;
  const auto f2 = y2 * (e2 + this->_b) - s * alph1 * k12 - alph2 * k22;
  const auto V1 = alph1 + s * (alph2 - V);
  return V1 * f1 + V * f2 + 0.5 * (V1 * V1) * k11 + 0.5 * (V * V) * k22 +
         s * V * V1 * k12;
}

Float SMO::kernel_function(const FloatArray& x1, const FloatArray& x2) const {
  if (this->_kernel_type == KernelType::Linear) {
    auto r = xt::linalg::dot(x1, xt::transpose(x2));
    return xt::linalg::dot(x1, xt::transpose(x2))(0);
  } else if (this->_kernel_type == KernelType::RBF) {
    FloatArray x2_t = xt::transpose(x2);
    auto s = xt::linalg::dot(x1, xt::transpose(x1)) +
             xt::linalg::dot(x2, x2_t) - 2 * xt::linalg::dot(x1, x2_t);
    return xt::exp(-s / (2 * std::pow(this->_sigma, 2)))(0);
  } else {
    throw std::invalid_argument("We only support Linear and RBF kernels !");
  }
}

Float SMO::clip_value(const Float value, const Float high, const Float low) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

std::int8_t SMO::take_step(const std::size_t i1, const std::size_t i2,
                           const FloatArray& x, const FloatArray& y,
                           const Float& y2, const Float& alph2,
                           const Float& e2) {
  if (i1 == i2) return 0;

  Float alph1 = this->_alphas[i1];
  Float y1 = y(i1);

  Float e1 = this->_errors[i1];
  if ((alph1 < this->_tol) || (alph1 > (this->_C - this->_tol))) {
    auto filtered_idxs =
        xt::flatten_indices(xt::argwhere(xt::not_equal(this->_alphas, 0)));
    FloatArray x_filtered = xt::view(x, xt::keep(filtered_idxs), xt::all());
    FloatArray y_filtered = xt::filter(y, xt::not_equal(this->_alphas, 0));
    FloatArray alphas_filtered =
        xt::filter(this->_alphas, xt::not_equal(this->_alphas, 0));
    e1 = this->eval(x_filtered, y_filtered, alphas_filtered,
                    xt::view(x, i1, xt::all())) -
         y1;
  }

  auto s = y1 * y2;

  Float L = 0;
  Float H = 0;

  if (y1 != y2) {
    L = std::max(0.0, alph2 - alph1);
    H = std::min(this->_C, this->_C + alph2 - alph1);
  } else {
    L = std::max(0.0, alph2 + alph1 - this->_C);
    H = std::min(this->_C, alph2 + alph1);
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
    a2 = this->clip_value(alph2 + y2 * (e1 - e2) / eta, H, L);
  } else {
    const auto Lobj =
        this->compute_gamma(alph1, alph2, L, k11, k12, k22, s, y1, y2, e1, e2);
    const auto Hobj =
        this->compute_gamma(alph1, alph2, H, k11, k12, k22, s, y1, y2, e1, e2);
    if (Lobj < (Hobj - this->_tol))
      a2 = L;
    else if (Lobj > (Hobj + this->_tol))
      a2 = H;
    else
      a2 = alph2;
  }

  if (std::abs(a2 - alph2) < this->_tol * (a2 + alph2 + this->_tol)) {
    return 0;
  }

  const auto a1 = alph1 + s * (alph2 - a2);
  auto new_b =
      this->compute_b(e1, e2, y1, a1, alph1, y2, a2, alph2, k11, k12, k22);
  auto delta_b = new_b - this->_b;
  this->_b = new_b;

  // Error cache.
  auto t1 = y1 * (a1 - alph1);
  auto t2 = y2 * (a2 - alph2);

  for (std::size_t idx = 0; idx < this->_alphas.size(); ++idx) {
    if ((this->_alphas[idx] > 0) && (this->_alphas[idx] < this->_C)) {
      this->_errors[idx] +=
          t1 * this->kernel_function(xt::view(x, i1, xt::all()),
                                     xt::view(x, idx, xt::all())) +
          t2 * this->kernel_function(xt::view(x, i2, xt::all()),
                                     xt::view(x, idx, xt::all())) -
          delta_b;
    }
  }

  this->_errors(i1) = 0.0;
  this->_errors(i2) = 0.0;

  this->_alphas(i1) = a1;
  this->_alphas(i2) = a2;

  if (this->_kernel_type == KernelType::Linear) {
    this->_w =
        this->compute_w(xt::view(x, i1, xt::all()), xt::view(x, i2, xt::all()),
                        y1, y2, a1, a2, alph1, alph2);
  }

  return 1;
}

FloatArray SMO::alphas() const { return this->_alphas; }
