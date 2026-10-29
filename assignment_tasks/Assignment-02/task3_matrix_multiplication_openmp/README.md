# task3_matrix_multiplication_openmp

- **Assignment**: Assignment-02
- **Task Description**: Parallel dense matrix multiplication using OpenMP for loops
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task3_matrix_mult.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task3_matrix_mult.c -o task3_matrix_mult -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task3_matrix_mult
```
