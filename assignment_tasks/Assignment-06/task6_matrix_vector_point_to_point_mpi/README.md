# task6_matrix_vector_point_to_point_mpi

- **Assignment**: Assignment-06
- **Task Description**: Parallel matrix-vector multiplication using point-to-point messaging
- **Primary Library / API**: MPI
- **Primary Code File**: `task6.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task6.c -o task6 -lm

# Execute across 4 processes
mpirun -np 4 ./task6
```
