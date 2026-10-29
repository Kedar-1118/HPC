# task3_loop_decomposition_comparison_openmp

- **Assignment**: Assignment-04
- **Task Description**: Evaluation of row-wise, column-wise, and block matrix decomposition strategies
- **Primary Library / API**: OpenMP
- **Primary Code File**: `decomposition_comparison.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp decomposition_comparison.cpp -o decomposition_comparison

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./decomposition_comparison
```
