# task1b_matrix_multiplication_openmp

- **Assignment**: Assignment-08
- **Task Description**: Shared-memory parallel matrix multiplication with OpenMP
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task1b.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task1b.c -o task1b -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task1b
```
