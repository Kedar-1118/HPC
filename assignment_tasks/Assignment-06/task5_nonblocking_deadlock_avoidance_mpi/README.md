# task5_nonblocking_deadlock_avoidance_mpi

- **Assignment**: Assignment-06
- **Task Description**: Non-blocking communication (MPI_Isend/MPI_Irecv) and deadlock resolution
- **Primary Library / API**: MPI
- **Primary Code File**: `task5.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task5.c -o task5 -lm

# Execute across 4 processes
mpirun -np 4 ./task5
```
