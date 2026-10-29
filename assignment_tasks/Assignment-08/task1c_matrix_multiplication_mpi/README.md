# task1c_matrix_multiplication_mpi

- **Assignment**: Assignment-08
- **Task Description**: Distributed-memory parallel matrix multiplication with MPI
- **Primary Library / API**: MPI
- **Primary Code File**: `task1c.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task1c.c -o task1c -lm

# Execute across 4 processes
mpirun -np 4 ./task1c
```
