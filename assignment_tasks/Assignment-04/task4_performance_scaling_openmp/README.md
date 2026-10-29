# task4_performance_scaling_openmp

- **Assignment**: Assignment-04
- **Task Description**: Performance scalability and Amdahl's Law speedup benchmarking
- **Primary Library / API**: OpenMP
- **Primary Code File**: `performance.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp performance.cpp -o performance

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./performance
```
