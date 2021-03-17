#include "ado/utils/logger_buffer.h"

namespace ado {
namespace utils {

LoggerBuffer::LoggerBuffer(const LoggerBuffer& buffer)
    : _log(buffer._log), _level(buffer._level) {
  this->_string_stream << buffer._string_stream.rdbuf();
}

LoggerBuffer::LoggerBuffer(
    const LogLevel& level,
    std::function<void(const std::string&, LogLevel)>
        log_function)
    : _level(level), _log(log_function) {}

LoggerBuffer::~LoggerBuffer() {
  this->_log(this->_string_stream.str(), this->_level);
}

}  // namespace utils
}  // namespace ado
