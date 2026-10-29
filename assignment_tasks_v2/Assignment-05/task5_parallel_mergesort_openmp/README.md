# task5_parallel_mergesort_openmp

- **Assignment**: Assignment-05
- **Task Description**: Parallel divide-and-conquer merge sort using task spawning and taskwait
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task5_parallel_mergesort.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task5_parallel_mergesort.cpp -o task5_parallel_mergesort

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task5_parallel_mergesort
```
