#ifndef ADO_UTILS_LOGGER_BUFFER_H
#define ADO_UTILS_LOGGER_BUFFER_H

#include <functional>
#include <sstream>

#include "ado/utils/logger_handler.h"

namespace ado {
namespace utils {

class LoggerBuffer {
 public:
  LoggerBuffer(const LoggerBuffer& buffer);

  LoggerBuffer(const LogLevel& level,
               std::function<void(const std::string&, LogLevel)>
                   log_function);

  ~LoggerBuffer();

  template <typename T>
  LoggerBuffer& operator<<(const T& value) {
    this->_string_stream << value;
    return *this;
  }

 private:
  std::stringstream _string_stream;
  std::function<void(const std::string&, LogLevel)>
      _log;
  LogLevel _level = LogLevel::Debug;
};

}  // namespace utils
}  // namespace ado

#endif  // AOD_UTILS_LOGGER_BUFFER_H
