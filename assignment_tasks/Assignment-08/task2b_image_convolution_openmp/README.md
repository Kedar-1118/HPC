# task2b_image_convolution_openmp

- **Assignment**: Assignment-08
- **Task Description**: 2D image convolution filter parallelized with OpenMP
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task2b.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with GCC with OpenMP support
gcc -O3 -fopenmp task2b.c -o task2b -lm

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task2b
```
