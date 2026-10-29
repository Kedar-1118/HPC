# task4_loop_scheduling_openmp

- **Assignment**: Assignment-02
- **Task Description**: Loop scheduling policies comparison (static, dynamic, guided with chunk sizes)
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task4_scheduling.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task4_scheduling.c -o task4_scheduling -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task4_scheduling
```
