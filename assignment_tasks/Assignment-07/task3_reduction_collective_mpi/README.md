# task3_reduction_collective_mpi

- **Assignment**: Assignment-07
- **Task Description**: Global reduction operations across distributed nodes using MPI_Reduce and MPI_Allreduce
- **Primary Library / API**: MPI
- **Primary Code File**: `task3.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task3.c -o task3 -lm

# Execute across 4 processes
mpirun -np 4 ./task3
```
