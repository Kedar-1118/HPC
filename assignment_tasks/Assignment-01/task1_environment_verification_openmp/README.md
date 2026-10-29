# task1_environment_verification_openmp

- **Assignment**: Assignment-01
- **Task Description**: OpenMP runtime environment verification and thread rank identification
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task1_hello.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task1_hello.c -o task1_hello -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task1_hello
```
