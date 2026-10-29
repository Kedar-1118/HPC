# task2_array_sum_reduction_openmp

- **Assignment**: Assignment-02
- **Task Description**: Parallel array sum reduction and thread synchronization
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task2_sum.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task2_sum.c -o task2_sum -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task2_sum
```
