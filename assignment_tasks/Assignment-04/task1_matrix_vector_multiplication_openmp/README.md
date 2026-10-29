# task1_matrix_vector_multiplication_openmp

- **Assignment**: Assignment-04
- **Task Description**: Dense matrix-vector multiplication with row-wise parallel decomposition
- **Primary Library / API**: OpenMP
- **Primary Code File**: `matrix_vector.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp matrix_vector.cpp -o matrix_vector

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./matrix_vector
```
