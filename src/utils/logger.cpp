#include "ado/utils/logger.h"

#include <string>

#include "ado/utils/logger_buffer.h"

namespace ado {
namespace utils {

Logger::Logger() {}

Logger& Logger::get() {
  static Logger logger;
  return logger;
}

void Logger::register_handler(std::unique_ptr<LoggerHandler> handler) {
  this->_handlers.emplace_back(std::move(handler));
}

void Logger::log(const std::string& message, const LogLevel level) {
  this->_mutex.lock();
  for (auto handler = this->_handlers.begin(); handler != this->_handlers.end();
       ++handler) {
    (*handler)->log(message, level);
  }
  this->_mutex.unlock();
}

}  // namespace utils
}  // namespace ado
