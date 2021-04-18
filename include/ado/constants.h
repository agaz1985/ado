#ifndef ADO_CONSTANTS_H
#define ADO_CONSTANTS_H

#include <limits.h>

#include "types.h"

namespace ado {

constexpr auto MAX_FLOAT_VALUE = std::numeric_limits<Float>::max();
constexpr auto MIN_FLOAT_VALUE = std::numeric_limits<Float>::min();
constexpr auto EPS_FLOAT_VALUE = std::numeric_limits<Float>::epsilon();

}  // namespace ado

#endif  // ADO_CONSTANTS_H