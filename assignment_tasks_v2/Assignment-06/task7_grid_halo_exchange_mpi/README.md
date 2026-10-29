# task7_grid_halo_exchange_mpi

- **Assignment**: Assignment-06
- **Task Description**: 2D Cartesian grid domain decomposition and boundary ghost/halo exchange
- **Primary Library / API**: MPI
- **Primary Code File**: `task7.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task7.c -o task7 -lm

# Execute across 4 processes
mpirun -np 4 ./task7
```
