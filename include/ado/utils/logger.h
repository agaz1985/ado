#ifndef ADO_UTILS_LOGGER_H
#define ADO_UTILS_LOGGER_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ado/utils/logger_buffer.h"
#include "ado/utils/logger_handler.h"

namespace ado {
namespace utils {

class Logger {
 public:
  static Logger& get();

  void register_handler(std::unique_ptr<LoggerHandler> handler);

  LoggerBuffer operator<<(const LogLevel& type) {
    return LoggerBuffer(type,
                        std::bind(&Logger::log, this, std::placeholders::_1,
                                  std::placeholders::_2, std::placeholders::_3,
                                  std::placeholders::_4));
  }

 private:
  Logger();
  Logger(const Logger&) = delete;
  Logger(const Logger&&) = delete;
  Logger& operator=(const Logger&) = delete;

  void log(const std::string& message, const LogLevel level,
           const std::string& file, const std::string& line);

  std::vector<std::unique_ptr<LoggerHandler>> _handlers;
  std::mutex _mutex;
};

}  // namespace utils
}  // namespace ado

#endif  // ADO_UTILS_LOGGER_H