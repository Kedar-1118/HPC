# task2_matrix_matrix_multiplication_openmp

- **Assignment**: Assignment-04
- **Task Description**: Dense matrix-matrix multiplication with multi-threaded loop collapsing
- **Primary Library / API**: OpenMP
- **Primary Code File**: `matrix_matrix.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp matrix_matrix.cpp -o matrix_matrix

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./matrix_matrix
```
