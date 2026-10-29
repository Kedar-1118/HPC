# task2_array_squaring_openmp

- **Assignment**: Assignment-01
- **Task Description**: Parallel square computation (0..999) and speedup benchmarking
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task2_square.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task2_square.c -o task2_square -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task2_square
```
