cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 14)

project(ado VERSION 1.0.0 DESCRIPTION "ado library")

# Project includes and src.
set(ADO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ADO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(ADO_HEADERS ${ADO_INCLUDE_DIR}/{CMAKE_PROJECT_NAME}/smo.h)
set(ADO_SOURCES ${ADO_SOURCE_DIR}/smo.cpp)

# Third-party libraries.
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

# Set defines.
add_definitions(-DHAVE_CBLAS=1)

# Create the library.
add_library(${CMAKE_PROJECT_NAME} SHARED ${ADO_SOURCES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${ADO_INCLUDE_DIR} ${xtensor_INCLUDE_DIRS})

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES SOVERSION 1)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${ADO_HEADERS})

# Link libraries.
target_link_libraries(${CMAKE_PROJECT_NAME} xtensor xtensor-blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} cblas)

# Unit Tests.
#enable_testing()
#add_subdirectory(test)

# Examples.
add_subdirectory(examples)

# Install options.
include(GNUInstallDirs)
install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
