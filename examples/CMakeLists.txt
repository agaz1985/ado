cmake_minimum_required(VERSION 3.19)
project(ado_examples)

# SVM Fitting.
add_executable(svm_fitting svm_fitting)
target_include_directories(svm_fitting PRIVATE ${ADO_INCLUDE_DIR})
target_link_libraries(svm_fitting ado)

