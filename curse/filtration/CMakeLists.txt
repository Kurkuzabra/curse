cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0057 NEW)

FIND_PACKAGE(Python COMPONENTS Interpreter Development)
FIND_PACKAGE(pybind11 CONFIG REQUIRED)
FIND_PACKAGE(Python3 COMPONENTS Interpreter NumPy)

SET(SOURCE_LIB graph_func.cpp)
SET(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ version selection")
SET(CMAKE_CXX_FLAGS_INIT "-lpthread -shared -std=c++20 -O4 -fPIC $(python3 -m pybind11 --includes)")

IF (APPLE)
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
    IF ((${ARCHITECTURE} STREQUAL "arm64") OR (${ARCHITECTURE} STREQUAL "arm64e"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $(python3-config --embed --ldflags)")
    ENDIF()
ENDIF()

pybind11_add_module(graph_func graph_func.cpp)
set_target_properties(graph_func PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../")
set_target_properties(graph_func PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../")