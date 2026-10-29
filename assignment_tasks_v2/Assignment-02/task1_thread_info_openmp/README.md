# task1_thread_info_openmp

- **Assignment**: Assignment-02
- **Task Description**: Thread count, thread IDs, dynamic threads, and nested parallelism query
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task1_thread_info.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task1_thread_info.c -o task1_thread_info -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task1_thread_info
```
