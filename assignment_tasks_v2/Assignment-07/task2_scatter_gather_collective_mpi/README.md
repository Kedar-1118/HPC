# task2_scatter_gather_collective_mpi

- **Assignment**: Assignment-07
- **Task Description**: Data partitioning and aggregation using MPI_Scatter and MPI_Gather
- **Primary Library / API**: MPI
- **Primary Code File**: `task2.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task2.c -o task2 -lm

# Execute across 4 processes
mpirun -np 4 ./task2
```
