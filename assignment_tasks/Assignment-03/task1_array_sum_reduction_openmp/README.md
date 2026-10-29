# task1_array_sum_reduction_openmp

- **Assignment**: Assignment-03
- **Task Description**: Array sum reduction comparing critical, atomic, and reduction clauses
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task1_array_sum.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task1_array_sum.cpp -o task1_array_sum

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task1_array_sum
```
