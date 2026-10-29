# task4_ai_gemm_matrix_multiplication_openmp

- **Assignment**: Assignment-03
- **Task Description**: AI GEMM general matrix multiplication benchmark with cache blocking
- **Primary Library / API**: OpenMP
- **Primary Code File**: `challenge1.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp challenge1.cpp -o challenge1

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./challenge1
```
