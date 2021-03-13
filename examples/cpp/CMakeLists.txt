cmake_minimum_required(VERSION 3.5)

project(ado_examples)

set(ADO_SVM_FITTING_EXAMPLE svm_synthetic_data.cpp)
add_executable(svm_synthetic_data ${ADO_SVM_FITTING_EXAMPLE})
target_link_libraries(svm_synthetic_data ado)

file(COPY ../data DESTINATION ${CMAKE_BINARY_DIR}/examples)
file(COPY ../python DESTINATION ${CMAKE_BINARY_DIR}/examples)
