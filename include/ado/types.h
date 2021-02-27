#ifndef ADO_TYPES_H
#define ADO_TYPES_H

#include <xtensor/xarray.hpp>

namespace ado {

using Float = std::double_t;
using FloatArray = xt::xarray<Float>;

using SizeArray = xt::xarray<std::size_t>;

}  // namespace ado

#endif  // ADO_TYPES_H