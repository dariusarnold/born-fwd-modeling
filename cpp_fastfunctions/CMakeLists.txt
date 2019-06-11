cmake_minimum_required(VERSION 3.10)
project(fastfunctions CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

# set the correct python version to use
set(PYBIND11_PYTHON_VERSION 3.6)
find_package(pybind11 REQUIRED)

pybind11_add_module(fastfunctions fastfunctions.cpp)

# add warnings
target_compile_options(fastfunctions PRIVATE -Wall -Wextra -pedantic -Werror)
# optimize
target_compile_options(fastfunctions PRIVATE -O3)