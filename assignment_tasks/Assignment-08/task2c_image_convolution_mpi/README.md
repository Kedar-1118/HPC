# task2c_image_convolution_mpi

- **Assignment**: Assignment-08
- **Task Description**: 2D image convolution filter parallelized with MPI halo exchanges
- **Primary Library / API**: MPI
- **Primary Code File**: `task2c.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task2c.c -o task2c -lm

# Execute across 4 processes
mpirun -np 4 ./task2c
```
