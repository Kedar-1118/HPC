# task7_block_vs_row_decomposition_mpi

- **Assignment**: Assignment-07
- **Task Description**: 2D Block Decomposition vs 1D Row-wise Decomposition showing cache hierarchy superiority
- **Primary Library / API**: MPI
- **Primary Code File**: `task7.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task7.c -o task7 -lm

# Execute across 4 processes
mpirun -np 4 ./task7
```
