#ifndef ADO_UTILS_IO_H
#define ADO_UTILS_IO_H

#include "ado/types.h"

namespace ado {
namespace utils {

using ado::FloatTensor;

FloatTensor load_data(const std::string& filepath);
void save_data(const FloatTensor& data, const std::string& filepath);

}  // namespace utils
}  // namespace ado

#endif  // ADO_UTILS_IO_H