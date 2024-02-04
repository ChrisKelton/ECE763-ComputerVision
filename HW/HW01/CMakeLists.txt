cmake_minimum_required(VERSION 3.10)

project (
    HW01
    VERSION 1.0
    LANGUAGES CXX
)

# add_library(UtilsCMakeExample utils.cpp)
add_library(EMExample expectation_maximization.cpp utils.cpp utils.h)

# add_executable(UtilsExample utils.cpp)
add_executable(EM expectation_maximization.cpp)

# target_link_libraries(UtilsExample PRIVATE UtilsCMakeExample)
target_link_libraries(EM PRIVATE EMExample)