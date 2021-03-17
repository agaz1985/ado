#ifndef ADO_UTILS_LOG_HANDLER_H
#define ADO_UTILS_LOG_HANDLER_H

#include <fstream>
#include <iostream>
#include <string>

namespace ado {
namespace utils {

enum class LogLevel { Debug = 0, Info = 1, Error = 2 };

class LoggerHandler {
 public:
  LoggerHandler(const LogLevel level);
  virtual ~LoggerHandler() = default;

  void log(const std::string& message, const LogLevel level);

  void set_level(const LogLevel level);

 protected:
  virtual void log_message(const std::string& message) = 0;
  virtual void log_error(const std::string& message) = 0;

 private:
  LogLevel _level = LogLevel::Debug;
};

class LogFileHandler : public LoggerHandler {
 public:
  LogFileHandler(const std::string& filepath_cout,
                 const std::string& filepath_cerr, const LogLevel level);

  ~LogFileHandler();

 protected:
  void virtual log_message(const std::string& message);
  void virtual log_error(const std::string& message);

 private:
  std::ofstream _log_cout;
  std::ofstream _log_cerr;
};

class LogStreamHandler : public LoggerHandler {
 public:
  LogStreamHandler(const LogLevel level);

 protected:
  void virtual log_message(const std::string& message);
  void virtual log_error(const std::string& message);
};

}  // namespace utils
}  // namespace ado

#endif  // ADO_UTILS_LOGGER_HANDLER_H