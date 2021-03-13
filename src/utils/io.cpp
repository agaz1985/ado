#include "ado/utils/io.h"

#include <fstream>
#include <ostream>
#include <string>
#include <xtensor/xcsv.hpp>

namespace ado {
namespace utils {

FloatArray load_data(const std::string& filepath) {
  std::fstream input_file(filepath.c_str(), std::ios::in);
  if (!input_file) {
    throw std::runtime_error("File does not exist !");
  } else {
    return xt::load_csv<Float>(input_file);
  }
}

void save_data(const FloatArray& data, const std::string& filepath) {
  std::ofstream output_file;
  output_file.open(filepath);
  xt::dump_csv(output_file, data);
}

}  // namespace utils
}  // namespace ado
