cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

project(ado VERSION 1.0.0 DESCRIPTION "ado library" LANGUAGES CXX)

set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Project includes and src.
set(ADO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(ADO_SOURCE_FILES
    ${ADO_SOURCE_DIR}/kernel.cpp
    ${ADO_SOURCE_DIR}/svm.cpp
)

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
add_library(${PROJECT_NAME} SHARED ${ADO_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src ${xtensor_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)

# Link libraries.
target_link_libraries(${CMAKE_PROJECT_NAME} xtensor xtensor-blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} cblas)

# Unit Tests.
#enable_testing()
#add_subdirectory(test)

# Examples.
add_subdirectory(examples)

# Post-build command for bulding bindings.
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME} 
    POST_BUILD
    WORKING_DIRECTORY ../bindings/python 
    COMMAND ./build.sh
)

# Install options.
install(TARGETS ${PROJECT_NAME} EXPORT AdoConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT AdoConfig DESTINATION share/Ado/cmake)

export(TARGETS ${PROJECT_NAME} FILE AdoConfig.cmake)
