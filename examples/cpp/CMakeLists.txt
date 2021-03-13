cmake_minimum_required(VERSION 3.5)

project(ado_examples)

set(ADO_SVM_FITTING_EXAMPLE svm_fitting.cpp)
add_executable(svm_fitting ${ADO_SVM_FITTING_EXAMPLE})
target_link_libraries(svm_fitting ado)

file(COPY ../data DESTINATION ${CMAKE_BINARY_DIR}/examples)
file(COPY ../python DESTINATION ${CMAKE_BINARY_DIR}/examples)
