# task1_broadcast_collective_mpi

- **Assignment**: Assignment-07
- **Task Description**: One-to-all broadcast communication using tree-based MPI_Bcast
- **Primary Library / API**: MPI
- **Primary Code File**: `task1.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task1.c -o task1 -lm

# Execute across 4 processes
mpirun -np 4 ./task1
```
