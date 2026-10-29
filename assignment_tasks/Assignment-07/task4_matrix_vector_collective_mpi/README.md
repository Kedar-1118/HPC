# task4_matrix_vector_collective_mpi

- **Assignment**: Assignment-07
- **Task Description**: Parallel matrix-vector multiplication using MPI_Scatter and MPI_Allgather
- **Primary Library / API**: MPI
- **Primary Code File**: `task4_mpi.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task4_mpi.c -o task4_mpi -lm

# Execute across 4 processes
mpirun -np 4 ./task4_mpi
```
