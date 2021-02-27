#include "ado/model.h"
#include "ado/types.h"

enum class KernelType { Linear = 0, RBF = 1 };

class SMO : public Model {
 public:
  SMO(const Float C = 1.0, const Float tol = 1e-4,
      const KernelType kernel_type = KernelType::Linear, const Float sigma = 5,
      const std::size_t max_steps = 1e3, const std::size_t seed = 16);

  // TODO: Add empty constructor, copy constructor and destructor, inheritance.

  void fit(const FloatArray& x, const FloatArray& y) override;
  FloatArray fit_predict(const FloatArray& x, const FloatArray& y) override;
  FloatArray predict(const FloatArray& x) override;

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
  KernelType _kernel_type = KernelType::Linear;
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
