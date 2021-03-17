#include <ctime>
#include <exception>
#include <iomanip>
#include <map>
#include <string>

#include "ado/utils/logger_buffer.h"

namespace {
using ado::utils::LogLevel;

const std::map<LogLevel, std::string> LogLevelMap = {
    {LogLevel::Debug, "DEBUG"},
    {LogLevel::Info, "INFO"},
    {LogLevel::Error, "ERROR"}};
}  // namespace

namespace ado {
namespace utils {

// LoggerHandler

LoggerHandler::LoggerHandler(const LogLevel level) : _level(level){};

void LoggerHandler::log(const std::string& message, const LogLevel level) {
  std::stringstream formatted_message;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  formatted_message << std::put_time(&tm, "%d-%m-%YT%H:%M:%S") << " - "
                    << LogLevelMap.find(level)->second << " - " << message;

  switch (level) {
    case LogLevel::Debug: {
      if (this->_level == LogLevel::Debug) {
        this->log_message(formatted_message.str());
      }
      break;
    }
    case LogLevel::Info: {
      if (this->_level != LogLevel::Error) {
        this->log_message(formatted_message.str());
      }
      break;
    }
    case LogLevel::Error: {
      this->log_error(formatted_message.str());
      break;
    }
    default:
      break;
  }
};

void LoggerHandler::set_level(const LogLevel level) { this->_level = level; }

// LogFileHandler

LogFileHandler::LogFileHandler(const std::string& filepath_cout,
                               const std::string& filepath_cerr,
                               const LogLevel level)
    : LoggerHandler(level), _log_cout(filepath_cout), _log_cerr(filepath_cerr) {
  if (!this->_log_cout.is_open()) {
    throw std::invalid_argument("Unable to open log file " + filepath_cout);
  }
  if (!this->_log_cerr.is_open()) {
    throw std::invalid_argument("Unable to open log file " + filepath_cerr);
  }
};

LogFileHandler::~LogFileHandler() {
  if (this->_log_cout.is_open()) {
    this->_log_cout.close();
  }
  if (this->_log_cerr.is_open()) {
    this->_log_cerr.close();
  }
}

void LogFileHandler::log_message(const std::string& message) {
  if (this->_log_cout.is_open()) {
    this->_log_cout << message << '\n';
    this->_log_cout.flush();
  }
}

void LogFileHandler::log_error(const std::string& message) {
  if (this->_log_cerr.is_open()) {
    this->_log_cerr << message << '\n';
    this->_log_cerr.flush();
  }
}

// LogStreamHandler

LogStreamHandler::LogStreamHandler(const LogLevel level)
    : LoggerHandler(level){};

void LogStreamHandler::log_message(const std::string& message) {
  std::cout << message << std::endl;
}

void LogStreamHandler::log_error(const std::string& message) {
  std::cerr << message << std::endl;
}

}  // namespace utils
}  // namespace ado
