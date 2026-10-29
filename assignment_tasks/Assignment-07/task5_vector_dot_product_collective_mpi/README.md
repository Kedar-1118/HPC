# task5_vector_dot_product_collective_mpi

- **Assignment**: Assignment-07
- **Task Description**: Parallel vector dot product using MPI_Scatter and MPI_Reduce
- **Primary Library / API**: MPI
- **Primary Code File**: `task5_mpi.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task5_mpi.c -o task5_mpi -lm

# Execute across 4 processes
mpirun -np 4 ./task5_mpi
```
