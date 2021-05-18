#include "ado/random/seed.h"

namespace ado {
namespace random {
void seed(const std::size_t seed_value) {
  xt::random::seed(seed_value);
}
}  // namespace random
}  // namespace ado
