#ifndef ADO_UTILS_IO_H
#define ADO_UTILS_IO_H

#include "ado/types.h"

namespace ado {
namespace utils {

using ado::FloatArray;

FloatArray load_data(const std::string& filepath);
void save_data(const FloatArray& data, const std::string& filepath);

}  // namespace utils
}  // namespace ado

#endif  // ADO_UTILS_IO_H