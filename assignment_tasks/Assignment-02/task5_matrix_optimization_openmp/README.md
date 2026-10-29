# task5_matrix_optimization_openmp

- **Assignment**: Assignment-02
- **Task Description**: Optimized matrix multiplication with loop collapsing and cache tuning
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task5_optimization.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task5_optimization.c -o task5_optimization -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task5_optimization
```
