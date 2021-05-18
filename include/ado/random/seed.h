#ifndef ADO_RANDOM_SEED_H
#define ADO_RANDOM_SEED_H

#include <ctime>
#include <xtensor/xrandom.hpp>

namespace ado {
namespace random {

/**
 * @brief Set the random seed to the specified value.
 *
 * @param seed_value random seed init value.
 */
void seed(const std::size_t seed_value = time(NULL));

}  // namespace random
}  // namespace ado

#endif  // ADO_RANDOM_SEED_H