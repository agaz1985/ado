#ifndef ADO_CORE_SVM_H
#define ADO_CORE_SVM_H

#include <memory>

#include "ado/core/kernel.h"
#include "ado/core/model.h"
#include "ado/types.h"

namespace ado {
namespace core {

/**
 * @brief Support Vector Machine (SVM) model.
 *
 * Implementation of a binary Support Vector Machine (SVM) based on the
 * Sequential Minimal Optimization (SMO) algorithm used for solving the
 * quadratic programming (QP) problem generated when training an SVM.
 * The SMO implementation is based on:
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

  /**
   * @brief Fit the model.
   *
   * @param x multi-dimensional array containing the training data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @param y array containing the target labels. The array must have shape
   * (N,1) or (N) and binary values [-1, 1]. With N number of samples.
   */
  void fit(const FloatArray& x, const FloatArray& y) override;

  /**
   * @brief Fit the model and run inference.
   *
   * @param x multi-dimensional array containing the training data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @param y array containing the target labels. The array must have shape
   * (N,1) or (N) and binary values [-1, 1]. With N number of samples.
   * @return FloatArray array containing the predicted labels. The array has
   * shape (N) and binary values [-1, 1]. With N number of samples.
   */
  FloatArray fit_predict(const FloatArray& x, const FloatArray& y) override;

  /**
   * @brief Run inference and return the predicted labels.
   *
   * @param x multi-dimensional array containing the input data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @return FloatArray array containing the predicted labels. The array has
   * shape (N) and binary values [-1, 1]. With N number of samples.
   */
  FloatArray predict(const FloatArray& x) override;

  /**
   * @brief Run inference and return the un-thresholded predicted values.
   *
   * @param x multi-dimensional array containing the input data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @return FloatArray array containing the un-thresholded predicted values.
   * The array has shape (N) and real values. With N number of
   * samples.
   */
  FloatArray decision_function(const FloatArray& x) override;

 private:
  /**
   * @brief Evaluate the model.
   */
  Float eval(const FloatArray& x, const FloatArray& y, const FloatArray& alphas,
             const FloatArray& xi) const;

  /**
   * @brief Compute the bias term.
   */
  Float compute_b(const Float& e1, const Float& e2, const Float& y1,
                  const Float& a1, const Float& alph1, const Float& y2,
                  const Float& a2, const Float& alph2, const Float& k11,
                  const Float& k12, const Float& k22) const;
  /**
   * @brief Compute the gamma term.
   */
  Float compute_gamma(const Float& alph1, const Float& alph2, const Float& V,
                      const Float& k11, const Float& k12, const Float& k22,
                      const Float& s, const Float& y1, const Float& y2,
                      const Float& e1, const Float& e2) const;
  /**
   * @brief Evaluate the kernel function.
   */
  Float kernel_function(const FloatArray& x1, const FloatArray& x2) const;

  /**
   * @brief Examine example step of the SMO algorithm.
   */
  std::int8_t examine_example(const std::size_t i2, const FloatArray& x,
                              const FloatArray& y);

  /**
   * @brief Take step of the SMO algorithm.
   */
  std::int8_t take_step(const std::size_t i1, const std::size_t i2,
                        const FloatArray& x, const FloatArray& y,
                        const Float& y2, const Float& alph2, const Float& e2);

  Float _C = 1.0;
  Float _tol = 1e-3;
  std::unique_ptr<Kernel> _kernel =
      std::make_unique<KernelPolynomial>(1.0, 1.0, 0.0);
  FloatArray _alphas = FloatArray();
  Float _b = 0.0;
  FloatArray _errors = FloatArray();
  FloatArray _x_support = FloatArray();
  FloatArray _y_support = FloatArray();
  std::size_t _max_steps = 1e3;
  std::size_t _seed = 16;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_SVM_H
