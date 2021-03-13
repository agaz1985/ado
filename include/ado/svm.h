#ifndef ADO_SVM_H
#define ADO_SVM_H

#include <memory>

#include "ado/kernel.h"
#include "ado/model.h"
#include "ado/types.h"

namespace ado {

/**
 * @brief Support Vector Machine (SVM) model.
 *
 * Implementation of a binary Support Vector Machine (SVM) based on the
 * Sequential Minimal Optimization (SMO) algorithm used for solving the
 * quadratic programming (QP) problem generated during training. The SMO
 * implementation is based on:
 * Platt, John. "Sequential minimal optimization: A fast algorithm for training
 * support vector machines." (1998).
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
 *
 */
class SVM : public Model {
 public:
  /**
   * @brief Construct a new SVM object
   *
   * @param C strictly positive regularization parameter.
   * @param tol tolerance for stopping criteria.
   * @param kernel kernel object (e.g. linear or rbf).
   * @param max_steps maximum number of iteration of the SMO algorithm.
   * @param seed used for the generation of pseudo random numbers and shuffling.
   */
  SVM(const Float C, const Float tol, std::unique_ptr<Kernel> kernel,
      const std::size_t max_steps, const std::size_t seed);

  SVM() = default;

  // TODO: Add empty constructor, copy constructor and destructor, inheritance.

  void fit(const FloatArray& x, const FloatArray& y) override;
  FloatArray fit_predict(const FloatArray& x, const FloatArray& y) override;
  FloatArray predict(const FloatArray& x) override;
  FloatArray decision_function(const FloatArray& x) override;

  FloatArray alphas() const;

 private:
  Float eval(const FloatArray& x, const FloatArray& y, const FloatArray& alphas,
             const FloatArray& xi) const;

  Float compute_b(const Float& e1, const Float& e2, const Float& y1,
                  const Float& a1, const Float& alph1, const Float& y2,
                  const Float& a2, const Float& alph2, const Float& k11,
                  const Float& k12, const Float& k22) const;

  FloatArray compute_w(const FloatArray& x1, const FloatArray& x2,
                       const FloatArray& y1, const FloatArray& y2,
                       const FloatArray& a1, const FloatArray& a2,
                       const FloatArray& alph1, const FloatArray& alph2) const;

  Float compute_gamma(const Float& alph1, const Float& alph2, const Float& V,
                      const Float& k11, const Float& k12, const Float& k22,
                      const Float& s, const Float& y1, const Float& y2,
                      const Float& e1, const Float& e2) const;

  Float kernel_function(const FloatArray& x1, const FloatArray& x2) const;

  static Float clip_value(const Float value, const Float high, const Float low);

  std::int8_t examine_example(const std::size_t i2, const FloatArray& x,
                              const FloatArray& y);

  std::int8_t take_step(const std::size_t i1, const std::size_t i2,
                        const FloatArray& x, const FloatArray& y,
                        const Float& y2, const Float& alph2, const Float& e2);

  Float _C = 1.0;
  Float _tol = 1e-3;
  std::unique_ptr<Kernel> _kernel = std::make_unique<KernelLinear>();
  Float _sigma = 1.0;
  FloatArray _alphas = FloatArray();
  Float _b = 0.0;
  FloatArray _w = FloatArray();
  FloatArray _errors = FloatArray();
  FloatArray _x_support = FloatArray();
  FloatArray _y_support = FloatArray();
  std::size_t _max_steps = 1e3;
  std::size_t _seed = 16;
};

}  // namespace ado

#endif  // ADO_SVM_H
