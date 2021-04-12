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
  void fit(const FloatTensor& x, const FloatTensor& y) override;

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
  FloatTensor fit_predict(const FloatTensor& x, const FloatTensor& y) override;

  /**
   * @brief Run inference and return the predicted labels.
   *
   * @param x multi-dimensional array containing the input data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @return FloatArray array containing the predicted labels. The array has
   * shape (N) and binary values [-1, 1]. With N number of samples.
   */
  FloatTensor predict(const FloatTensor& x) override;

  /**
   * @brief Run inference and return the un-thresholded predicted values.
   *
   * @param x multi-dimensional array containing the input data. The array
   * must have shape (N,M), with N number of samples, and M number of features.
   * @return FloatArray array containing the un-thresholded predicted values.
   * The array has shape (N) and real values. With N number of
   * samples.
   */
  FloatTensor decision_function(const FloatTensor& x) override;

 private:
  /**
   * @brief Evaluate the model.
   */
  Float eval(const FloatTensor& x, const FloatTensor& y,
             const FloatTensor& alphas, const FloatTensor& xi) const;

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
  Float kernel_function(const FloatTensor& x1, const FloatTensor& x2) const;

  /**
   * @brief Examine example step of the SMO algorithm.
   */
  std::int8_t examine_example(const std::size_t i2, const FloatTensor& x,
                              const FloatTensor& y);

  /**
   * @brief Take step of the SMO algorithm.
   */
  std::int8_t take_step(const std::size_t i1, const std::size_t i2,
                        const FloatTensor& x, const FloatTensor& y,
                        const Float& y2, const Float& alph2, const Float& e2);

  Float C_ = 1.0;
  Float tol_ = 1e-3;
  std::unique_ptr<Kernel> kernel_ =
      std::make_unique<KernelPolynomial>(1.0, 1.0, 0.0);
  FloatTensor alphas_ = FloatTensor();
  Float b_ = 0.0;
  FloatTensor errors_ = FloatTensor();
  FloatTensor x_support_ = FloatTensor();
  FloatTensor y_support_ = FloatTensor();
  std::size_t max_steps_ = 1e3;
};

}  // namespace core
}  // namespace ado

#endif  // ADO_CORE_SVM_H
