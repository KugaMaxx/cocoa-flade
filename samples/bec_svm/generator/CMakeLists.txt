cmake_minimum_required(VERSION 3.5.1)
project(generator)

# Configurable features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configurable features
set(CMAKE_BUILD_TYPE "DEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Find OpenCV supports
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Pybind11 supports
find_package(pybind11 REQUIRED)

pybind11_add_module(
    generator_ops
    MODULE generator_ops.cpp
)

target_link_libraries(
    generator_ops
    PRIVATE ${OpenCV_LIBS}
)
