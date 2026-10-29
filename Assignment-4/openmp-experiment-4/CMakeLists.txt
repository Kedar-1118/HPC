cmake_minimum_required(VERSION 3.10)
project(OpenMP_Experiment4 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)

# Output binaries to bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Executables ----

# Task 1: Matrix-Vector Multiplication
add_executable(matrix_vector src/matrix_vector.cpp)
target_link_libraries(matrix_vector OpenMP::OpenMP_CXX)

# Task 2: Matrix-Matrix Multiplication
add_executable(matrix_matrix src/matrix_matrix.cpp)
target_link_libraries(matrix_matrix OpenMP::OpenMP_CXX)

# Task 3: Decomposition Comparison
add_executable(decomposition_comparison src/decomposition_comparison.cpp)
target_link_libraries(decomposition_comparison OpenMP::OpenMP_CXX)

# Task 4: Performance Analysis
add_executable(performance src/performance.cpp)
target_link_libraries(performance OpenMP::OpenMP_CXX)

# Challenge 3: Recursive Tasks (Merge Sort)
add_executable(recursive_tasks src/recursive_tasks.cpp)
target_link_libraries(recursive_tasks OpenMP::OpenMP_CXX)

# Create results directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results/plots)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
