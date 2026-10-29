# task6_collective_performance_benchmark_mpi

- **Assignment**: Assignment-07
- **Task Description**: Empirical scaling benchmark of collective primitives vs message size and process count
- **Primary Library / API**: MPI
- **Primary Code File**: `task6.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task6.c -o task6 -lm

# Execute across 4 processes
mpirun -np 4 ./task6
```
