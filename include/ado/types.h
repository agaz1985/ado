#ifndef ADO_TYPES_H
#define ADD_TYPES_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

namespace ado {

#ifdef ADO_DOUBLE_PRECISION
using Float = std::double_t;
using Int = std::int64_t;
#else
using Float = std::float_t;
using Int = std::int32_t;
#endif

template <typename T>
using Tensor = xt::xarray<T>;

using FloatTensor = Tensor<Float>;
using IntTensor = Tensor<Int>;

template <typename T>
using TensorShape = typename Tensor<T>::shape_type;

using SizeArray = Tensor<std::size_t>;

}  // namespace ado

#endif  // ADO_TYPES_H